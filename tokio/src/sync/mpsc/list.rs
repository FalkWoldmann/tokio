//! A concurrent, lock-free, FIFO list.
//!
//! # Block-liveness invariant
//!
//! Almost every `unsafe` block in this module and in [`block`] dereferences a
//! `*const Block<T>` or `NonNull<Block<T>>`, and each such proof cites the following
//! module invariant, referred to below as the *block-liveness invariant*:
//!
//! > Every block pointer reachable from `Tx::block_tail`, from `Rx::head`, from
//! > `Rx::free_head`, or from any `BlockHeader::next` chain starting at one of those,
//! > points to a live, initialized `Block<T>` that was produced by `Block::new` and
//! > converted with `Box::into_raw`. Such a pointer is never dangling and is never
//! > aliased by a `&mut Block<T>`.
//!
//! It holds because of how blocks are created and destroyed:
//!
//! * Blocks enter the list only through `Block::new`, either in `channel()` or in
//!   `Block::grow`/`Tx::reclaim_block`, and are published by a `Release`
//!   compare-exchange on `next` (or on `block_tail`). A thread that reaches a block by
//!   an `Acquire` load of `next`/`block_tail` therefore sees a fully initialized block.
//! * Blocks leave the list only through the receiver, of which there is exactly one:
//!   `Rx::reclaim_blocks` (which hands the block to `Tx::reclaim_block`, where it is
//!   either recycled back into the list or freed) and `Rx::free_blocks` at teardown.
//! * `Rx::reclaim_blocks` unlinks a block only once `observed_tail_position()` returns
//!   `Some`, i.e. once the `RELEASED` bit is visible. That bit is published by
//!   `Block::tx_release`, whose own contract is that no sender will access the block
//!   again. So by the time the receiver can free a block, no sender can reach it.
//! * The single `&mut Block<T>` in the module, `Block::reclaim` via
//!   `Tx::reclaim_block`, is taken only on a block the receiver has already unlinked.
//!
//! Where a proof below needs more than this — an ordering edge, or exclusivity against
//! the *receiver* rather than the senders — it says so explicitly.

use crate::loom::sync::atomic::{AtomicPtr, AtomicUsize};
use crate::loom::thread;
use crate::sync::mpsc::block::{self, Block};

use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};

/// List queue transmit handle.
pub(crate) struct Tx<T> {
    /// Tail in the `Block` mpmc list.
    block_tail: AtomicPtr<Block<T>>,

    /// Position to push the next message. This references a block and offset
    /// into the block.
    tail_position: AtomicUsize,
}

/// List queue receive handle
pub(crate) struct Rx<T> {
    /// Pointer to the block being processed.
    head: NonNull<Block<T>>,

    /// Next slot index to process.
    index: usize,

    /// Pointer to the next block pending release.
    free_head: NonNull<Block<T>>,
}

/// Return value of `Rx::try_pop`.
pub(crate) enum TryPopResult<T> {
    /// Successfully popped a value.
    Ok(T),
    /// The channel is empty.
    ///
    /// Note that `list.rs` only tracks the close state set by senders. If the
    /// channel is closed by `Rx::close()`, then `TryPopResult::Empty` is still
    /// returned, and the close state needs to be handled by `chan.rs`.
    Empty,
    /// The channel is empty and closed.
    ///
    /// Returned when the send half is closed (all senders dropped).
    Closed,
    /// The channel is not empty, but the first value is being written.
    Busy,
}

pub(crate) fn channel<T>() -> (Tx<T>, Rx<T>) {
    // Create the initial block shared between the tx and rx halves.
    let initial_block = Block::new(0);
    let initial_block_ptr = Box::into_raw(initial_block);

    let tx = Tx {
        block_tail: AtomicPtr::new(initial_block_ptr),
        tail_position: AtomicUsize::new(0),
    };

    let head = NonNull::new(initial_block_ptr).unwrap();

    let rx = Rx {
        head,
        index: 0,
        free_head: head,
    };

    (tx, rx)
}

impl<T> Tx<T> {
    /// Pushes a value into the list.
    pub(crate) fn push(&self, value: T) {
        // First, claim a slot for the value. `Acquire` is used here to
        // synchronize with the `fetch_add` in `reclaim_blocks`.
        let slot_index = self.tail_position.fetch_add(1, Acquire);

        // Load the current block and write the value
        let block = self.find_block(slot_index);

        // SAFETY:
        // Two operations.
        // 1. `block.as_ref()` — creating `&Block<T>`. Contract: aligned, initialized,
        //    live for the returned lifetime, no `&mut` alias. Discharged by the
        //    block-liveness invariant: `find_block` returns a pointer it reached by
        //    walking `block_tail`/`next` with `Acquire` loads, or one it just allocated.
        // 2. `Block::write(slot_index, value)` — contract: the slot is empty, no
        //    concurrent access to it, and `slot_index` belongs to this block.
        //    Evidence: `slot_index` was claimed by this thread's `fetch_add` on
        //    `tail_position`, so it is unique to this call — no other sender can target
        //    the same slot, and the receiver reads the slot only after `write` publishes
        //    its ready bit. `find_block(slot_index)` returns precisely the block whose
        //    `start_index` matches, so the index is in range, and a freshly allocated or
        //    reclaimed block has all ready bits clear, so the slot is empty.
        // Postcondition: per `Block::write`, the block may be read and then freed by the
        // receiver from this point on, so `block` must not be used again — and it is not.
        unsafe {
            // Write the value to the block
            block.as_ref().write(slot_index, value);
        }
    }

    /// Closes the send half of the list.
    ///
    /// Similar process as pushing a value, but instead of writing the value &
    /// setting the ready flag, the `TX_CLOSED` flag is set on the block.
    pub(crate) fn close(&self) {
        // First, claim a slot for the value. This is the last slot that will be
        // claimed.
        let slot_index = self.tail_position.fetch_add(1, Acquire);

        let block = self.find_block(slot_index);

        // SAFETY:
        // Two operations.
        // 1. `block.as_ref()` — as in `push` above, discharged by the block-liveness
        //    invariant.
        // 2. `Block::tx_close()` — that method imposes no memory-safety obligation (its
        //    body is a single atomic `fetch_or`); its protocol obligation is that the
        //    caller is the sender half and no further sender will write to the block.
        //    `close` is called once, by `Chan`'s drop path, after this thread has claimed
        //    the final `slot_index` with `fetch_add`, so no sender will claim it again.
        unsafe { block.as_ref().tx_close() }
    }

    fn find_block(&self, slot_index: usize) -> NonNull<Block<T>> {
        // The start index of the block that contains `index`.
        let start_index = block::start_index(slot_index);

        // The index offset into the block
        let offset = block::offset(slot_index);

        // Load the current head of the block
        let mut block_ptr = self.block_tail.load(Acquire);

        // SAFETY:
        // Operation: creating `&Block<T>` from `block_ptr`.
        // Contract: non-null, aligned, pointing to an initialized `Block<T>` that is live
        // for the returned lifetime, with no `&mut` alias.
        // Evidence: `block_ptr` was just loaded from `block_tail` with `Acquire`. By the
        // block-liveness invariant that pointer is never null and never dangling: the
        // receiver frees a block only after `tx_release` has declared that no sender will
        // reach it, and `block_tail` never points at such a block (advancing `block_tail`
        // past a block is what triggers its `tx_release` in the first place). The
        // `Acquire` load synchronizes with the `Release` compare-exchange that published
        // the pointer, so the block's contents are visible.
        // Only `header.start_index` is read through this reference, via `distance`.
        let block = unsafe { &*block_ptr };

        // Calculate the distance between the tail ptr and the target block
        let distance = block.distance(start_index);

        // Decide if this call to `find_block` should attempt to update the
        // `block_tail` pointer.
        //
        // Updating `block_tail` is not always performed in order to reduce
        // contention.
        //
        // When set, as the routine walks the linked list, it attempts to update
        // `block_tail`. If the update cannot be performed, `try_updating_tail`
        // is unset.
        let mut try_updating_tail = distance > offset;

        // Walk the linked list of blocks until the block with `start_index` is
        // found.
        loop {
            // SAFETY:
            // Operation: creating `&Block<T>` from `block_ptr`.
            // Contract and evidence as for the `block_tail` dereference above, extended to
            // the walk: on later iterations `block_ptr` comes from `next_block`, which is
            // either an `Acquire` `load_next` (so the block was published by a `Release`
            // CAS on `next`) or a block `Block::grow` just allocated and linked. Both are
            // covered by the block-liveness invariant, and the sender holds no other
            // reference that could be `&mut`.
            // This reference is used only for `is_at_index`, `load_next`, `is_final`,
            // `grow` and `tx_release`, all of which take `&self`.
            let block = unsafe { &(*block_ptr) };

            if block.is_at_index(start_index) {
                // SAFETY:
                // Contract from `NonNull::new_unchecked`: the pointer must be non-null.
                // Evidence: `block_ptr` was dereferenced successfully just above under the
                // block-liveness invariant, which states these pointers are never null; it
                // originates either from the `Acquire` load of `block_tail` (a field that
                // is initialized non-null in `channel()` and only ever CAS'd to another
                // block pointer) or from `next_block.as_ptr()`, which came from a
                // `NonNull`.
                return unsafe { NonNull::new_unchecked(block_ptr) };
            }

            let next_block = block
                .load_next(Acquire)
                // There is no allocated next block, grow the linked list.
                .unwrap_or_else(|| block.grow());

            // If the block is **not** final, then the tail pointer cannot be
            // advanced any more.
            try_updating_tail &= block.is_final();

            if try_updating_tail {
                // Advancing `block_tail` must happen when walking the linked
                // list. `block_tail` may not advance passed any blocks that are
                // not "final". At the point a block is finalized, it is unknown
                // if there are any prior blocks that are unfinalized, which
                // makes it impossible to advance `block_tail`.
                //
                // While walking the linked list, `block_tail` can be advanced
                // as long as finalized blocks are traversed.
                //
                // Release ordering is used to ensure that any subsequent reads
                // are able to see the memory pointed to by `block_tail`.
                //
                // Acquire is not needed as any "actual" value is not accessed.
                // At this point, the linked list is walked to acquire blocks.
                if self
                    .block_tail
                    .compare_exchange(block_ptr, next_block.as_ptr(), Release, Relaxed)
                    .is_ok()
                {
                    // Synchronize with any senders
                    let tail_position = self.tail_position.fetch_add(0, Release);

                    // SAFETY:
                    // Contract from `Block::tx_release`: the block will no longer be
                    // accessed by any sender, from now on.
                    // Evidence: this thread just won the `compare_exchange` that advanced
                    // `block_tail` past `block`, so `block` is no longer reachable from
                    // `block_tail`. A sender reaches a block only by starting at
                    // `block_tail` and walking forwards, so no sender that starts after
                    // this CAS can reach it. Senders already past the CAS are accounted
                    // for by `tail_position`: the `fetch_add(0, Release)` above reads the
                    // current tail, and the receiver only frees the block once its
                    // `index` has caught up to that position, by which point those senders
                    // have finished writing. `try_updating_tail` also guarantees `block`
                    // is final (`is_final()` was checked), so every slot in it has been
                    // written.
                    // Postcondition: the receiver may now free `block`, so it must not be
                    // touched again — the loop advances to `next_block` immediately below.
                    unsafe {
                        block.tx_release(tail_position);
                    }
                } else {
                    // A concurrent sender is also working on advancing
                    // `block_tail` and this thread is falling behind.
                    //
                    // Stop trying to advance the tail pointer
                    try_updating_tail = false;
                }
            }

            block_ptr = next_block.as_ptr();

            thread::yield_now();
        }
    }

    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions are violated:
    ///
    /// - The `block` was created by [`Box::into_raw`].
    /// - The `block` is not currently part of any linked list.
    /// - The `block` is a valid pointer to a [`Block<T>`].
    pub(crate) unsafe fn reclaim_block(&self, mut block: NonNull<Block<T>>) {
        // The block has been removed from the linked list and ownership
        // is reclaimed.
        //
        // Before dropping the block, see if it can be reused by
        // inserting it back at the end of the linked list.
        //
        // First, reset the data
        //
        // Safety: caller guarantees the block is valid and not in any list.
        unsafe {
            block.as_mut().reclaim();
        }

        let mut reused = false;

        // Attempt to insert the block at the end
        //
        // Walk at most three times
        let curr_ptr = self.block_tail.load(Acquire);

        // The pointer can never be null
        debug_assert!(!curr_ptr.is_null());

        // Safety: curr_ptr is never null.
        let mut curr = unsafe { NonNull::new_unchecked(curr_ptr) };

        // TODO: Unify this logic with Block::grow
        for _ in 0..3 {
            // SAFETY:
            // Two operations.
            // 1. `curr.as_ref()` — creating `&Block<T>`. Discharged by the block-liveness
            //    invariant: `curr` starts at the `Acquire` load of `block_tail` and is
            //    then only reassigned to the `next` pointer returned by a failed
            //    `try_push`, both of which are links in the live list.
            // 2. `try_push(&mut block, ..)` — contract: `block` must not be freed until it
            //    has been removed from the list. Evidence: `block` is the reclaimed block
            //    this method owns; it is freed below only on the `!reused` path, i.e. only
            //    when no `try_push` succeeded and it was therefore never linked in.
            match unsafe { curr.as_ref().try_push(&mut block, AcqRel, Acquire) } {
                Ok(()) => {
                    reused = true;
                    break;
                }
                Err(next) => {
                    curr = next;
                }
            }
        }

        if !reused {
            // Safety:
            //
            // 1. Caller guarantees the block is valid and not in any list.
            // 2. The block was created by `Box::into_raw`.
            let _ = unsafe { Box::from_raw(block.as_ptr()) };
        }
    }
}

impl<T> fmt::Debug for Tx<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Tx")
            .field("block_tail", &self.block_tail.load(Relaxed))
            .field("tail_position", &self.tail_position.load(Relaxed))
            .finish()
    }
}

impl<T> Rx<T> {
    pub(crate) fn is_empty(&self, tx: &Tx<T>) -> bool {
        // SAFETY:
        // Operation: `self.head.as_ref()`, creating `&Block<T>`.
        // Contract: aligned, initialized, live for the returned lifetime, no `&mut` alias.
        // Evidence: by the block-liveness invariant `head` always points at a live block.
        // Exclusivity against the only party that frees blocks — the receiver — follows
        // from `&self` here: freeing happens in `reclaim_blocks`/`free_blocks`, both of
        // which take `&mut self`, so neither can run for the returned lifetime.
        let block = unsafe { self.head.as_ref() };
        if block.has_value(self.index) {
            return false;
        }

        // It is possible that a block has no value "now" but the list is still not empty.
        // To be sure, it is necessary to check the length of the list.
        self.len(tx) == 0
    }

    // Guaranteed to return true if `slot_index` is the fake message sent on channel close.
    // Guaranteed to return false if `slot_index` is a fully sent message.
    //
    // For messages that are partially sent, may return either true or false.
    fn is_maybe_closed(&self, tx: &Tx<T>, slot_index: usize) -> bool {
        let start_index = block::start_index(slot_index);

        let tail = tx.block_tail.load(Acquire);
        // SAFETY: Only the receiver frees blocks, so since we are the receiver, this will not be
        // freed right now.
        let tail_ref = unsafe { &*tail };
        if tail_ref.is_at_index(start_index) {
            return !tail_ref.has_value(slot_index);
        }

        // This method is optimized for checking whether the last value is present, so most of the
        // time it is in `block_tail`. However, this isn't always the case since it's possible
        // that the list was grown with an empty block, in which case `block_tail` points one block
        // too far. To handle this case, we walk the list from the head.
        let mut block_ptr = Some(self.head);

        while let Some(block) = block_ptr {
            // SAFETY: Only the receiver frees blocks, so since we are the receiver, this will not
            // be freed right now.
            let block_ref = unsafe { block.as_ref() };
            if block_ref.is_at_index(start_index) {
                return !block_ref.has_value(slot_index);
            }
            block_ptr = block_ref.load_next(Acquire);
        }
        true
    }

    pub(crate) fn len(&self, tx: &Tx<T>) -> usize {
        let tail_position = tx.tail_position.load(Acquire);
        let mut len = tail_position.wrapping_sub(self.index);
        debug_assert!(0 <= len as isize);
        if len == 0 {
            return 0;
        }
        // There are messages present in the queue. However, it's possible that the last message is
        // a fake "closed" message that we do not wish to count. To avoid counting it, we do not
        // count the last message if the ready bit is unset.
        //
        // Note that it is also possible for the ready bit to be unset on a normal message, but
        // this happens only if that message is currently being sent *right now* in parallel on
        // another thread. That is okay because it is optional to count messages that are currently
        // being sent.
        if self.is_maybe_closed(tx, tail_position.wrapping_sub(1)) {
            len -= 1;
        }
        len
    }

    /// Pops the next value off the queue.
    pub(crate) fn pop(&mut self, tx: &Tx<T>) -> Option<block::Read<T>> {
        // Advance `head`, if needed
        if !self.try_advancing_head() {
            return None;
        }

        self.reclaim_blocks(tx);

        // SAFETY:
        // Two operations.
        // 1. `self.head.as_ref()` — creating `&Block<T>`. Discharged by the block-liveness
        //    invariant. `try_advancing_head` above returned `true`, so `head` is the block
        //    containing `self.index`, and `reclaim_blocks` only frees blocks strictly
        //    before `head`, so `head` itself is still live.
        // 2. `Block::read(self.index)` — contract: no concurrent access to the slot, and
        //    `self.index` belongs to this block.
        //    Evidence: this is `&mut self` on the sole `Rx`, and `Block::read` is called
        //    nowhere else, so no other reader exists; senders only ever *write* slots they
        //    have exclusively claimed, and never the one being read, because a slot is
        //    readable only once its ready bit is published. `try_advancing_head`
        //    established that `head.is_at_index(block::start_index(self.index))`, so the
        //    index is in range.
        // Postcondition: `Block::read` moves the value out of the slot, so the slot must
        // never be read again. `self.index` is advanced past it on exactly the
        // `Read::Value` path, which is the only path that took ownership of a `T`.
        unsafe {
            let block = self.head.as_ref();

            let ret = block.read(self.index);

            if let Some(block::Read::Value(..)) = ret {
                self.index = self.index.wrapping_add(1);
            }

            ret
        }
    }

    /// Pops the next value off the queue, detecting whether the block
    /// is busy or empty on failure.
    ///
    /// This function exists because `Rx::pop` can return `None` even if the
    /// channel's queue contains a message that has been completely written.
    /// This can happen if the fully delivered message is behind another message
    /// that is in the middle of being written to the block, since the channel
    /// can't return the messages out of order.
    pub(crate) fn try_pop(&mut self, tx: &Tx<T>) -> TryPopResult<T> {
        let tail_position = tx.tail_position.load(Acquire);
        let result = self.pop(tx);

        match result {
            Some(block::Read::Value(t)) => TryPopResult::Ok(t),
            Some(block::Read::Closed) => TryPopResult::Closed,
            None if tail_position == self.index => TryPopResult::Empty,
            None => TryPopResult::Busy,
        }
    }

    /// Tries advancing the block pointer to the block referenced by `self.index`.
    ///
    /// Returns `true` if successful, `false` if there is no next block to load.
    fn try_advancing_head(&mut self) -> bool {
        let block_index = block::start_index(self.index);

        loop {
            let next_block = {
                // SAFETY:
                // Operation: `self.head.as_ref()`, creating `&Block<T>`.
                // Contract: aligned, initialized, live, no `&mut` alias.
                // Evidence: by the block-liveness invariant `head` points at a live block.
                // `head` only ever moves forwards, to a block returned by an `Acquire`
                // `load_next`, and `reclaim_blocks` frees only blocks strictly before
                // `head` (it stops when `free_head == head`), so `head` is never freed.
                // The reference is confined to this inner scope and is dropped before
                // `self.head` is reassigned, so the `&mut self` here does not alias it.
                let block = unsafe { self.head.as_ref() };

                if block.is_at_index(block_index) {
                    return true;
                }

                block.load_next(Acquire)
            };

            let next_block = match next_block {
                Some(next_block) => next_block,
                None => {
                    return false;
                }
            };

            self.head = next_block;

            thread::yield_now();
        }
    }

    fn reclaim_blocks(&mut self, tx: &Tx<T>) {
        while self.free_head != self.head {
            // SAFETY:
            // Three operations, all on `block == self.free_head`.
            // 1. `block.as_ref().observed_tail_position()` and 2. `.load_next(Relaxed)` —
            //    creating `&Block<T>`. Discharged by the block-liveness invariant:
            //    `free_head` trails `head` along the live list and is only advanced here.
            //    Nothing has freed this block yet, because this method is the only place
            //    that hands a block to `reclaim_block`, it runs on the sole receiver under
            //    `&mut self`, and it frees each block exactly once before advancing
            //    `free_head` past it.
            // 3. `tx.reclaim_block(block)` — contract: `block` was created by
            //    `Box::into_raw`, is a valid pointer to a `Block<T>`, and is not currently
            //    part of any linked list.
            //    Evidence: creation via `Box::into_raw` and validity are the
            //    block-liveness invariant. Not-in-any-list is the crux and rests on two
            //    facts established just above: `observed_tail_position()` returned `Some`,
            //    so `Block::tx_release` ran and no sender will access this block again
            //    (and `block_tail` has already been advanced past it); and
            //    `required_index <= self.index`, so the receiver has consumed every slot
            //    in it. `free_head` is advanced to `next_block` *before* the call, so the
            //    receiver no longer references it either. The `&Block<T>` references from
            //    (1) and (2) have been dropped by this point, so `reclaim_block`'s
            //    `&mut Block<T>` is unique.
            unsafe {
                // Get a handle to the block that will be freed and update
                // `free_head` to point to the next block.
                let block = self.free_head;

                let observed_tail_position = block.as_ref().observed_tail_position();

                let required_index = match observed_tail_position {
                    Some(i) => i,
                    None => return,
                };

                if required_index > self.index {
                    return;
                }

                // We may read the next pointer with `Relaxed` ordering as it is
                // guaranteed that the `reclaim_blocks` routine trails the `recv`
                // routine. Any memory accessed by `reclaim_blocks` has already
                // been acquired by `recv`.
                let next_block = block.as_ref().load_next(Relaxed);

                // Update the free list head
                self.free_head = next_block.unwrap();

                // Push the emptied block onto the back of the queue, making it
                // available to senders.
                tx.reclaim_block(block);
            }

            thread::yield_now();
        }
    }

    /// Effectively `Drop` all the blocks. Should only be called once, when
    /// the list is dropping.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// * This method is called at most once for a given `Rx`. Calling it twice would
    ///   free every block a second time. (Under `debug_assertions` the pointers are set
    ///   to `NonNull::dangling()` afterwards so that a second call trips the assertion,
    ///   but that check is absent in release builds.)
    /// * No sender can still reach any block in the list — i.e. every `Tx` has been
    ///   dropped — since this frees blocks that `Tx::block_tail` may still name.
    /// * All values still held in the blocks have already been drained, or their
    ///   destructors are intentionally skipped: this frees the blocks' backing memory
    ///   without dropping any `T` still stored in a slot.
    pub(super) unsafe fn free_blocks(&mut self) {
        debug_assert_ne!(self.free_head, NonNull::dangling());

        let mut cur = Some(self.free_head);

        #[cfg(debug_assertions)]
        {
            // to trigger the debug assert above so as to catch that we
            // don't call `free_blocks` more than once.
            self.free_head = NonNull::dangling();
            self.head = NonNull::dangling();
        }

        while let Some(block) = cur {
            // SAFETY:
            // Operation: `block.as_ref()`, creating a short-lived `&Block<T>` to read
            // `next`.
            // Contract: aligned, initialized, live, no `&mut` alias.
            // Evidence: by the block-liveness invariant every block on the chain starting
            // at `free_head` is live; the ones freed so far in this loop are strictly
            // before `block`, and each is freed only after its `next` has been read. By
            // this method's `# Safety` precondition no sender remains, and `&mut self`
            // makes this the only receiver access. `Relaxed` suffices because all senders
            // are gone, so there is no concurrent writer to synchronize with.
            cur = unsafe { block.as_ref() }.load_next(Relaxed);
            // SAFETY:
            // Contract from `Box::from_raw`: non-null, aligned, pointing to a valid
            // `Block<T>` allocated by the global allocator with the layout `Box` will free
            // it with, and the caller must relinquish all other access.
            // Evidence: allocation provenance and layout come from the block-liveness
            // invariant (`Block::new` allocates with `Layout::new::<Block<T>>()` and hands
            // the pointer to `Box::into_raw`). The `&Block<T>` from the line above has
            // been consumed by `load_next` and is dead here, `cur` already holds the
            // successor, and this method's precondition rules out any other holder.
            // Postcondition: the allocation is freed exactly once. Note this drops the
            // `Block<T>` itself, not any `T` still sitting in its slots — those are
            // `MaybeUninit` and have no drop glue; see this method's `# Safety` docs.
            drop(unsafe { Box::from_raw(block.as_ptr()) });
        }
    }
}

impl<T> fmt::Debug for Rx<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Rx")
            .field("head", &self.head)
            .field("index", &self.index)
            .field("free_head", &self.free_head)
            .finish()
    }
}
