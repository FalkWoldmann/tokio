#![cfg_attr(not(feature = "sync"), allow(dead_code, unreachable_pub))]

//! A one-shot channel is used for sending a single message between
//! asynchronous tasks. The [`channel`] function is used to create a
//! [`Sender`] and [`Receiver`] handle pair that form the channel.
//!
//! The `Sender` handle is used by the producer to send the value.
//! The `Receiver` handle is used by the consumer to receive the value.
//!
//! Each handle can be used on separate tasks.
//!
//! Since the `send` method is not async, it can be used anywhere. This includes
//! sending between two runtimes, and using it from non-async code.
//!
//! If the [`Receiver`] is closed before receiving a message which has already
//! been sent, the message will remain in the channel until the receiver is
//! dropped, at which point the message will be dropped immediately.
//!
//! # Examples
//!
//! ```
//! use tokio::sync::oneshot;
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! let (tx, rx) = oneshot::channel();
//!
//! tokio::spawn(async move {
//!     if let Err(_) = tx.send(3) {
//!         println!("the receiver dropped");
//!     }
//! });
//!
//! match rx.await {
//!     Ok(v) => println!("got = {:?}", v),
//!     Err(_) => println!("the sender dropped"),
//! }
//! # }
//! ```
//!
//! If the sender is dropped without sending, the receiver will fail with
//! [`error::RecvError`]:
//!
//! ```
//! use tokio::sync::oneshot;
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! let (tx, rx) = oneshot::channel::<u32>();
//!
//! tokio::spawn(async move {
//!     drop(tx);
//! });
//!
//! match rx.await {
//!     Ok(_) => panic!("This doesn't happen"),
//!     Err(_) => println!("the sender dropped"),
//! }
//! # }
//! ```
//!
//! To use a `oneshot` channel in a `tokio::select!` loop, add `&mut` in front of
//! the channel.
//!
//! ```
//! use tokio::sync::oneshot;
//! use tokio::time::{interval, sleep, Duration};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn _doc() {}
//! # #[tokio::main(flavor = "current_thread", start_paused = true)]
//! # async fn main() {
//! let (send, mut recv) = oneshot::channel();
//! let mut interval = interval(Duration::from_millis(100));
//!
//! # let handle =
//! tokio::spawn(async move {
//!     sleep(Duration::from_secs(1)).await;
//!     send.send("shut down").unwrap();
//! });
//!
//! loop {
//!     tokio::select! {
//!         _ = interval.tick() => println!("Another 100ms"),
//!         msg = &mut recv => {
//!             println!("Got message: {}", msg.unwrap());
//!             break;
//!         }
//!     }
//! }
//! # handle.await.unwrap();
//! # }
//! ```
//!
//! To use a `Sender` from a destructor, put it in an [`Option`] and call
//! [`Option::take`].
//!
//! ```
//! use tokio::sync::oneshot;
//!
//! struct SendOnDrop {
//!     sender: Option<oneshot::Sender<&'static str>>,
//! }
//! impl Drop for SendOnDrop {
//!     fn drop(&mut self) {
//!         if let Some(sender) = self.sender.take() {
//!             // Using `let _ =` to ignore send errors.
//!             let _ = sender.send("I got dropped!");
//!         }
//!     }
//! }
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn _doc() {}
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! let (send, recv) = oneshot::channel();
//!
//! let send_on_drop = SendOnDrop { sender: Some(send) };
//! drop(send_on_drop);
//!
//! assert_eq!(recv.await, Ok("I got dropped!"));
//! # }
//! ```

use crate::loom::cell::UnsafeCell;
use crate::loom::sync::atomic::AtomicUsize;
use crate::loom::sync::Arc;
#[cfg(all(tokio_unstable, feature = "tracing"))]
use crate::util::trace;

use std::fmt;
use std::future::Future;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::Ordering::{self, AcqRel, Acquire};
use std::task::Poll::{Pending, Ready};
use std::task::{ready, Context, Poll, Waker};

/// Sends a value to the associated [`Receiver`].
///
/// A pair of both a [`Sender`] and a [`Receiver`]  are created by the
/// [`channel`](fn@channel) function.
///
/// # Examples
///
/// ```
/// use tokio::sync::oneshot;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let (tx, rx) = oneshot::channel();
///
/// tokio::spawn(async move {
///     if let Err(_) = tx.send(3) {
///         println!("the receiver dropped");
///     }
/// });
///
/// match rx.await {
///     Ok(v) => println!("got = {:?}", v),
///     Err(_) => println!("the sender dropped"),
/// }
/// # }
/// ```
///
/// If the sender is dropped without sending, the receiver will fail with
/// [`error::RecvError`]:
///
/// ```
/// use tokio::sync::oneshot;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let (tx, rx) = oneshot::channel::<u32>();
///
/// tokio::spawn(async move {
///     drop(tx);
/// });
///
/// match rx.await {
///     Ok(_) => panic!("This doesn't happen"),
///     Err(_) => println!("the sender dropped"),
/// }
/// # }
/// ```
///
/// To use a `Sender` from a destructor, put it in an [`Option`] and call
/// [`Option::take`].
///
/// ```
/// use tokio::sync::oneshot;
///
/// struct SendOnDrop {
///     sender: Option<oneshot::Sender<&'static str>>,
/// }
/// impl Drop for SendOnDrop {
///     fn drop(&mut self) {
///         if let Some(sender) = self.sender.take() {
///             // Using `let _ =` to ignore send errors.
///             let _ = sender.send("I got dropped!");
///         }
///     }
/// }
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn _doc() {}
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let (send, recv) = oneshot::channel();
///
/// let send_on_drop = SendOnDrop { sender: Some(send) };
/// drop(send_on_drop);
///
/// assert_eq!(recv.await, Ok("I got dropped!"));
/// # }
/// ```
///
/// [`Option`]: std::option::Option
/// [`Option::take`]: std::option::Option::take
#[derive(Debug)]
pub struct Sender<T> {
    inner: Option<Arc<Inner<T>>>,
    #[cfg(all(tokio_unstable, feature = "tracing"))]
    resource_span: tracing::Span,
}

/// Receives a value from the associated [`Sender`].
///
/// A pair of both a [`Sender`] and a [`Receiver`]  are created by the
/// [`channel`](fn@channel) function.
///
/// This channel has no `recv` method because the receiver itself implements the
/// [`Future`] trait. To receive a `Result<T, `[`error::RecvError`]`>`, `.await` the `Receiver` object directly.
///
/// The `poll` method on the `Future` trait is allowed to spuriously return
/// `Poll::Pending` even if the message has been sent. If such a spurious
/// failure happens, then the caller will be woken when the spurious failure has
/// been resolved so that the caller can attempt to receive the message again.
/// Note that receiving such a wakeup does not guarantee that the next call will
/// succeed — it could fail with another spurious failure. (A spurious failure
/// does not mean that the message is lost. It is just delayed.)
///
/// [`Future`]: trait@std::future::Future
///
/// # Cancel safety
///
/// Awaiting a `&mut Receiver<T>` is cancel safe. If it is used as a branch in
/// [`tokio::select!`](crate::select) and another branch completes first, it is
/// guaranteed that no message was received on this
/// channel.
///
/// # Examples
///
/// ```
/// use tokio::sync::oneshot;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let (tx, rx) = oneshot::channel();
///
/// tokio::spawn(async move {
///     if let Err(_) = tx.send(3) {
///         println!("the receiver dropped");
///     }
/// });
///
/// match rx.await {
///     Ok(v) => println!("got = {:?}", v),
///     Err(_) => println!("the sender dropped"),
/// }
/// # }
/// ```
///
/// If the sender is dropped without sending, the receiver will fail with
/// [`error::RecvError`]:
///
/// ```
/// use tokio::sync::oneshot;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let (tx, rx) = oneshot::channel::<u32>();
///
/// tokio::spawn(async move {
///     drop(tx);
/// });
///
/// match rx.await {
///     Ok(_) => panic!("This doesn't happen"),
///     Err(_) => println!("the sender dropped"),
/// }
/// # }
/// ```
///
/// To use a `Receiver` in a `tokio::select!` loop, add `&mut` in front of the
/// channel.
///
/// ```
/// use tokio::sync::oneshot;
/// use tokio::time::{interval, sleep, Duration};
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn _doc() {}
/// # #[tokio::main(flavor = "current_thread", start_paused = true)]
/// # async fn main() {
/// let (send, mut recv) = oneshot::channel();
/// let mut interval = interval(Duration::from_millis(100));
///
/// # let handle =
/// tokio::spawn(async move {
///     sleep(Duration::from_secs(1)).await;
///     send.send("shut down").unwrap();
/// });
///
/// loop {
///     tokio::select! {
///         _ = interval.tick() => println!("Another 100ms"),
///         msg = &mut recv => {
///             println!("Got message: {}", msg.unwrap());
///             break;
///         }
///     }
/// }
/// # handle.await.unwrap();
/// # }
/// ```
#[derive(Debug)]
pub struct Receiver<T> {
    inner: Option<Arc<Inner<T>>>,
    #[cfg(all(tokio_unstable, feature = "tracing"))]
    resource_span: tracing::Span,
    #[cfg(all(tokio_unstable, feature = "tracing"))]
    async_op_span: tracing::Span,
    #[cfg(all(tokio_unstable, feature = "tracing"))]
    async_op_poll_span: tracing::Span,
}

pub mod error {
    //! `Oneshot` error types.

    use std::fmt;

    /// Error returned by the `Future` implementation for `Receiver`.
    ///
    /// This error is returned by the receiver when the sender is dropped without sending.
    #[derive(Debug, Eq, PartialEq, Clone)]
    pub struct RecvError(pub(super) ());

    /// Error returned by the `try_recv` function on `Receiver`.
    #[derive(Debug, Eq, PartialEq, Clone)]
    pub enum TryRecvError {
        /// The send half of the channel has not yet sent a value.
        Empty,

        /// The send half of the channel was dropped without sending a value.
        Closed,
    }

    // ===== impl RecvError =====

    impl fmt::Display for RecvError {
        fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(fmt, "channel closed")
        }
    }

    impl std::error::Error for RecvError {}

    // ===== impl TryRecvError =====

    impl fmt::Display for TryRecvError {
        fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                TryRecvError::Empty => write!(fmt, "channel empty"),
                TryRecvError::Closed => write!(fmt, "channel closed"),
            }
        }
    }

    impl std::error::Error for TryRecvError {}
}

use self::error::*;

struct Inner<T> {
    /// Manages the state of the inner cell.
    state: AtomicUsize,

    /// The value. This is set by `Sender` and read by `Receiver`. The state of
    /// the cell is tracked by `state`.
    value: UnsafeCell<Option<T>>,

    /// The task to notify when the receiver drops without consuming the value.
    ///
    /// ## Safety
    ///
    /// The `TX_TASK_SET` bit in the `state` field is set if this field is
    /// initialized. If that bit is unset, this field may be uninitialized.
    tx_task: Task,

    /// The task to notify when the value is sent.
    ///
    /// ## Safety
    ///
    /// The `RX_TASK_SET` bit in the `state` field is set if this field is
    /// initialized. If that bit is unset, this field may be uninitialized.
    rx_task: Task,
}

struct Task(UnsafeCell<MaybeUninit<Waker>>);

impl Task {
    /// # Safety
    ///
    /// The caller must ensure that, for the whole duration of this call:
    ///
    /// * the cell holds an initialized `Waker` (for `rx_task`/`tx_task` this means the
    ///   corresponding `RX_TASK_SET`/`TX_TASK_SET` state bit is set), and
    /// * no other thread is concurrently writing to or dropping that `Waker`, i.e. no
    ///   concurrent `set_task` or `drop_task` on the same cell. Concurrent `will_wake`
    ///   and `with_task` calls on the same cell are permitted, since both take only
    ///   shared access.
    ///
    /// The caller is responsible for establishing this through the `state` field's
    /// happens-before edges; the `Waker` itself is not synchronized by this type.
    unsafe fn will_wake(&self, cx: &mut Context<'_>) -> bool {
        // SAFETY:
        // Operation: the call to `with_task`.
        // Contract from `with_task`: initialized `Waker` in the cell and no concurrent
        // writer, for the duration of the call.
        // Evidence: this function's own `# Safety` precondition is exactly that contract,
        // and nothing happens between accepting it and making this call.
        unsafe { self.with_task(|w| w.will_wake(cx.waker())) }
    }

    /// # Safety
    ///
    /// Same obligations as [`Self::will_wake`]: the cell must hold an initialized `Waker`
    /// and no concurrent `set_task`/`drop_task` may run on it for the duration of the
    /// call. `f` receives a shared reference that is valid only until it returns; it must
    /// not be escaped.
    unsafe fn with_task<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Waker) -> R,
    {
        self.0.with(|ptr| {
            // SAFETY:
            // Justifies two operations:
            // 1. `(*ptr).as_ptr()` — dereferencing `ptr` to reach the `MaybeUninit<Waker>`.
            //    `UnsafeCell::with` yields a non-null, aligned pointer to the cell's
            //    contents, and `MaybeUninit::as_ptr` imposes no initialization
            //    requirement of its own; it only needs a valid place to project from.
            //    The `Inner` holding this cell outlives the call because `&self` borrows
            //    it.
            // 2. `&*waker` — creating `&Waker`.
            //    Contract: non-null, aligned, pointing to an initialized `Waker`, with no
            //    `&mut Waker` alias live over the returned lifetime.
            //    Evidence: non-null and aligned follow from (1); initialization is this
            //    function's `# Safety` precondition; exclusivity against a writer is also
            //    that precondition (no concurrent `set_task`/`drop_task`). Other
            //    concurrent readers only create further `&Waker`, which is permitted.
            //    The reference does not outlive this closure, and `f` cannot escape it
            //    because it is passed as an anonymous-lifetime `&Waker`.
            let waker: *const Waker = unsafe { (*ptr).as_ptr() };
            // SAFETY: operation (2) of the proof directly above.
            f(unsafe { &*waker })
        })
    }

    /// # Safety
    ///
    /// The caller must ensure that, for the whole duration of this call:
    ///
    /// * the cell holds an initialized `Waker` (the corresponding `*_TASK_SET` state bit
    ///   is set), and
    /// * the caller has *exclusive* access to that `Waker` — no concurrent `with_task`,
    ///   `will_wake`, `set_task`, or `drop_task` on the same cell.
    ///
    /// # Postcondition
    ///
    /// The `Waker` is dropped and the cell is left uninitialized. The caller must clear
    /// the corresponding `*_TASK_SET` state bit (or otherwise guarantee that nobody will
    /// observe it as set) before anyone may read the cell again, or the next reader will
    /// form a reference to a dropped value.
    unsafe fn drop_task(&self) {
        self.0.with_mut(|ptr| {
            // SAFETY:
            // Justifies two operations:
            // 1. `(*ptr).as_mut_ptr()` — `UnsafeCell::with_mut` yields a non-null, aligned
            //    pointer to the cell's contents, and `MaybeUninit::as_mut_ptr` requires
            //    only a valid place. The cell outlives the call via `&self`.
            // 2. `ptr.drop_in_place()` — contract from `ptr::drop_in_place`: the pointer
            //    must be non-null, properly aligned, valid for both reads and writes, and
            //    point to a valid (initialized) value that is not accessed by anything
            //    else while the destructor runs.
            //    Evidence: non-null and aligned from (1); initialization and exclusivity
            //    are this function's `# Safety` precondition. `Waker::drop` is
            //    std-provided and cannot re-enter this cell.
            // Postcondition: the cell is now uninitialized; see this function's docs.
            let ptr: *mut Waker = unsafe { (*ptr).as_mut_ptr() };
            // SAFETY: operation (2) of the proof directly above.
            unsafe {
                ptr.drop_in_place();
            }
        });
    }

    /// # Safety
    ///
    /// The caller must ensure that, for the whole duration of this call:
    ///
    /// * the cell does *not* hold an initialized `Waker` (the corresponding `*_TASK_SET`
    ///   state bit is clear), and
    /// * the caller has *exclusive* access to the cell — no concurrent `with_task`,
    ///   `will_wake`, `set_task`, or `drop_task` on it.
    ///
    /// The first condition is required for correctness rather than soundness: `ptr::write`
    /// does not drop the previous value, so overwriting an initialized slot would leak the
    /// old `Waker` rather than cause UB.
    ///
    /// # Postcondition
    ///
    /// The cell holds an initialized `Waker`. The caller must set the corresponding
    /// `*_TASK_SET` state bit so that the eventual `drop_task` runs, otherwise the `Waker`
    /// is leaked.
    unsafe fn set_task(&self, cx: &mut Context<'_>) {
        self.0.with_mut(|ptr| {
            // SAFETY:
            // Justifies two operations:
            // 1. `(*ptr).as_mut_ptr()` — as in `drop_task`: `UnsafeCell::with_mut` yields a
            //    non-null, aligned pointer to the cell, and `MaybeUninit::as_mut_ptr`
            //    needs no initialization.
            // 2. `ptr.write(..)` — contract from `ptr::write`: the destination must be
            //    non-null, properly aligned, and valid for writes. It explicitly does
            //    *not* require the destination to be initialized, and does not drop the
            //    old value.
            //    Evidence: non-null, aligned and valid for writes follow from (1) —
            //    `MaybeUninit<Waker>` has the same size and alignment as `Waker` (std
            //    docs) and the cell is live for the call. Exclusivity is this function's
            //    `# Safety` precondition, so no concurrent reader can observe the torn
            //    intermediate state.
            //    The value written is `cx.waker().clone()`, a fully owned `Waker`;
            //    ownership moves into the cell.
            // Postcondition: the cell is now initialized; see this function's docs.
            let ptr: *mut Waker = unsafe { (*ptr).as_mut_ptr() };
            // SAFETY: operation (2) of the proof directly above.
            unsafe {
                ptr.write(cx.waker().clone());
            }
        });
    }
}

#[derive(Clone, Copy)]
struct State(usize);

/// Creates a new one-shot channel for sending single values across asynchronous
/// tasks.
///
/// The function returns separate "send" and "receive" handles. The `Sender`
/// handle is used by the producer to send the value. The `Receiver` handle is
/// used by the consumer to receive the value.
///
/// Each handle can be used on separate tasks.
///
/// # Examples
///
/// ```
/// use tokio::sync::oneshot;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let (tx, rx) = oneshot::channel();
///
/// tokio::spawn(async move {
///     if let Err(_) = tx.send(3) {
///         println!("the receiver dropped");
///     }
/// });
///
/// match rx.await {
///     Ok(v) => println!("got = {:?}", v),
///     Err(_) => println!("the sender dropped"),
/// }
/// # }
/// ```
#[track_caller]
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    #[cfg(all(tokio_unstable, feature = "tracing"))]
    let resource_span = {
        let location = std::panic::Location::caller();

        let resource_span = tracing::trace_span!(
            parent: None,
            "runtime.resource",
            concrete_type = "Sender|Receiver",
            kind = "Sync",
            loc.file = location.file(),
            loc.line = location.line(),
            loc.col = location.column(),
        );

        resource_span.in_scope(|| {
            tracing::trace!(
            target: "runtime::resource::state_update",
            tx_dropped = false,
            tx_dropped.op = "override",
            )
        });

        resource_span.in_scope(|| {
            tracing::trace!(
            target: "runtime::resource::state_update",
            rx_dropped = false,
            rx_dropped.op = "override",
            )
        });

        resource_span.in_scope(|| {
            tracing::trace!(
            target: "runtime::resource::state_update",
            value_sent = false,
            value_sent.op = "override",
            )
        });

        resource_span.in_scope(|| {
            tracing::trace!(
            target: "runtime::resource::state_update",
            value_received = false,
            value_received.op = "override",
            )
        });

        resource_span
    };

    let inner = Arc::new(Inner {
        state: AtomicUsize::new(State::new().as_usize()),
        value: UnsafeCell::new(None),
        tx_task: Task(UnsafeCell::new(MaybeUninit::uninit())),
        rx_task: Task(UnsafeCell::new(MaybeUninit::uninit())),
    });

    let tx = Sender {
        inner: Some(inner.clone()),
        #[cfg(all(tokio_unstable, feature = "tracing"))]
        resource_span: resource_span.clone(),
    };

    #[cfg(all(tokio_unstable, feature = "tracing"))]
    let async_op_span = resource_span
        .in_scope(|| tracing::trace_span!("runtime.resource.async_op", source = "Receiver::await"));

    #[cfg(all(tokio_unstable, feature = "tracing"))]
    let async_op_poll_span =
        async_op_span.in_scope(|| tracing::trace_span!("runtime.resource.async_op.poll"));

    let rx = Receiver {
        inner: Some(inner),
        #[cfg(all(tokio_unstable, feature = "tracing"))]
        resource_span,
        #[cfg(all(tokio_unstable, feature = "tracing"))]
        async_op_span,
        #[cfg(all(tokio_unstable, feature = "tracing"))]
        async_op_poll_span,
    };

    (tx, rx)
}

impl<T> Sender<T> {
    /// Attempts to send a value on this channel, returning it back if it could
    /// not be sent.
    ///
    /// This method consumes `self` as only one value may ever be sent on a `oneshot`
    /// channel. It is not marked async because sending a message to a `oneshot`
    /// channel never requires any form of waiting.  Because of this, the `send`
    /// method can be used in both synchronous and asynchronous code without
    /// problems.
    ///
    /// A successful send occurs when it is determined that the other end of the
    /// channel has not hung up already. An unsuccessful send would be one where
    /// the corresponding receiver has already been deallocated. Note that a
    /// return value of `Err` means that the data will never be received, but
    /// a return value of `Ok` does *not* mean that the data will be received.
    /// It is possible for the corresponding receiver to hang up immediately
    /// after this function returns `Ok`.
    ///
    /// # Examples
    ///
    /// Send a value to another task
    ///
    /// ```
    /// use tokio::sync::oneshot;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (tx, rx) = oneshot::channel();
    ///
    /// tokio::spawn(async move {
    ///     if let Err(_) = tx.send(3) {
    ///         println!("the receiver dropped");
    ///     }
    /// });
    ///
    /// match rx.await {
    ///     Ok(v) => println!("got = {:?}", v),
    ///     Err(_) => println!("the sender dropped"),
    /// }
    /// # }
    /// ```
    pub fn send(mut self, t: T) -> Result<(), T> {
        let inner = self.inner.take().unwrap();

        // SAFETY:
        // Operation: writing `Some(t)` through the `value` `UnsafeCell`.
        // Required contract: this thread must have exclusive access to the cell for the
        // duration of the write, and the pointer must be valid for writes of `Option<T>`.
        // Evidence:
        // - `UnsafeCell::with_mut` yields a non-null, aligned pointer to the cell's
        //   contents, and the `Inner` outlives this call because `inner` is an owned
        //   handle to it.
        // - By the documented invariant on `VALUE_SENT`, the receiver accesses the cell
        //   only once that bit is set. The bit is set exclusively by `complete()`, which
        //   this method calls only *after* the write below, and `send` consumes `self`,
        //   so no other sender exists and `complete()` has not yet run. Therefore
        //   `VALUE_SENT` is clear and the receiver is not touching the cell.
        // - The previous contents are `None` (set in `Inner::new` and never written by
        //   anyone else before `complete()`), so the assignment's implicit drop of the old
        //   `Option<T>` is a no-op and cannot run arbitrary user code here.
        inner.value.with_mut(|ptr| unsafe {
            *ptr = Some(t);
        });

        if !inner.complete() {
            // SAFETY:
            // Operation: the call to `Inner::consume_value`.
            // Contract from `consume_value`: only one side may call it at a time; the
            // sender may call it while `VALUE_SENT` is clear, the receiver only once it is
            // set.
            // Evidence: `complete()` returned `false`, which per `State::set_complete`
            // happens exactly when the `CLOSED` bit was already set — in which case the
            // CAS loop breaks early and `VALUE_SENT` is *not* set, and never will be,
            // because `set_complete` is the only writer of that bit and `send` consumes
            // `self`. So the receiver will never be permitted to touch the cell, and this
            // sender has exclusive access.
            // Postcondition: the cell is left holding `None`; the value is returned to the
            // caller, so it is neither leaked nor dropped twice.
            return Err(unsafe { inner.consume_value() }.unwrap());
        }

        #[cfg(all(tokio_unstable, feature = "tracing"))]
        self.resource_span.in_scope(|| {
            tracing::trace!(
            target: "runtime::resource::state_update",
            value_sent = true,
            value_sent.op = "override",
            )
        });

        Ok(())
    }

    /// Waits for the associated [`Receiver`] handle to close.
    ///
    /// A [`Receiver`] is closed by either calling [`close`] explicitly or the
    /// [`Receiver`] value is dropped.
    ///
    /// This function is useful when paired with `select!` to abort a
    /// computation when the receiver is no longer interested in the result.
    ///
    /// # Return
    ///
    /// Returns a `Future` which must be awaited on.
    ///
    /// [`Receiver`]: Receiver
    /// [`close`]: Receiver::close
    ///
    /// # Examples
    ///
    /// Basic usage
    ///
    /// ```
    /// use tokio::sync::oneshot;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (mut tx, rx) = oneshot::channel::<()>();
    ///
    /// tokio::spawn(async move {
    ///     drop(rx);
    /// });
    ///
    /// tx.closed().await;
    /// println!("the receiver dropped");
    /// # }
    /// ```
    ///
    /// Paired with select
    ///
    /// ```
    /// use tokio::sync::oneshot;
    /// use tokio::time::{self, Duration};
    ///
    /// async fn compute() -> String {
    ///     // Complex computation returning a `String`
    /// # "hello".to_string()
    /// }
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (mut tx, rx) = oneshot::channel();
    ///
    /// tokio::spawn(async move {
    ///     tokio::select! {
    ///         _ = tx.closed() => {
    ///             // The receiver dropped, no need to do any further work
    ///         }
    ///         value = compute() => {
    ///             // The send can fail if the channel was closed at the exact same
    ///             // time as when compute() finished, so just ignore the failure.
    ///             let _ = tx.send(value);
    ///         }
    ///     }
    /// });
    ///
    /// // Wait for up to 10 seconds
    /// let _ = time::timeout(Duration::from_secs(10), rx).await;
    /// # }
    /// ```
    pub async fn closed(&mut self) {
        use std::future::poll_fn;

        #[cfg(all(tokio_unstable, feature = "tracing"))]
        let resource_span = self.resource_span.clone();
        #[cfg(all(tokio_unstable, feature = "tracing"))]
        let closed = trace::async_op(
            || poll_fn(|cx| self.poll_closed(cx)),
            resource_span,
            "Sender::closed",
            "poll_closed",
            false,
        );
        #[cfg(not(all(tokio_unstable, feature = "tracing")))]
        let closed = poll_fn(|cx| self.poll_closed(cx));

        closed.await;
    }

    /// Returns `true` if the associated [`Receiver`] handle has been dropped.
    ///
    /// A [`Receiver`] is closed by either calling [`close`] explicitly or the
    /// [`Receiver`] value is dropped.
    ///
    /// If `true` is returned, a call to `send` will always result in an error.
    ///
    /// [`Receiver`]: Receiver
    /// [`close`]: Receiver::close
    ///
    /// # Examples
    ///
    /// ```
    /// use tokio::sync::oneshot;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (tx, rx) = oneshot::channel();
    ///
    /// assert!(!tx.is_closed());
    ///
    /// drop(rx);
    ///
    /// assert!(tx.is_closed());
    /// assert!(tx.send("never received").is_err());
    /// # }
    /// ```
    pub fn is_closed(&self) -> bool {
        let inner = self.inner.as_ref().unwrap();

        let state = State::load(&inner.state, Acquire);
        state.is_closed()
    }

    /// Checks whether the `oneshot` channel has been closed, and if not, schedules the
    /// `Waker` in the provided `Context` to receive a notification when the channel is
    /// closed.
    ///
    /// A [`Receiver`] is closed by either calling [`close`] explicitly, or when the
    /// [`Receiver`] value is dropped.
    ///
    /// Note that on multiple calls to poll, only the `Waker` from the `Context` passed
    /// to the most recent call will be scheduled to receive a wakeup.
    ///
    /// [`Receiver`]: struct@crate::sync::oneshot::Receiver
    /// [`close`]: fn@crate::sync::oneshot::Receiver::close
    ///
    /// # Return value
    ///
    /// This function returns:
    ///
    ///  * `Poll::Pending` if the channel is still open.
    ///  * `Poll::Ready(())` if the channel is closed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tokio::sync::oneshot;
    ///
    /// use std::future::poll_fn;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (mut tx, mut rx) = oneshot::channel::<()>();
    ///
    /// tokio::spawn(async move {
    ///     rx.close();
    /// });
    ///
    /// poll_fn(|cx| tx.poll_closed(cx)).await;
    ///
    /// println!("the receiver dropped");
    /// # }
    /// ```
    pub fn poll_closed(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        ready!(crate::trace::trace_leaf());

        // Keep track of task budget
        let coop = ready!(crate::task::coop::poll_proceed(cx));

        let inner = self.inner.as_ref().unwrap();

        let mut state = State::load(&inner.state, Acquire);

        if state.is_closed() {
            coop.made_progress();
            return Ready(());
        }

        if state.is_tx_task_set() {
            // SAFETY:
            // Contract from `Task::will_wake`: the cell holds an initialized `Waker` and
            // no concurrent `set_task`/`drop_task` runs on it.
            // Evidence:
            // - Initialization: `TX_TASK_SET` is set in the `Acquire` load above, which by
            //   that bit's documented invariant means `tx_task` is initialized. The
            //   `Acquire` load synchronizes with the `AcqRel` `fetch_or` in `set_tx_task`,
            //   so the `Waker` written before it is visible here.
            // - Exclusivity against writers: `tx_task` is only ever written by the sender,
            //   and `poll_closed` takes `&mut self` on the `Sender`, so this is the only
            //   sender and no concurrent `set_task`/`drop_task` can run. The receiver only
            //   ever *reads* `tx_task` (in `close()`), which is a permitted concurrent
            //   shared access.
            let will_notify = unsafe { inner.tx_task.will_wake(cx) };

            if !will_notify {
                state = State::unset_tx_task(&inner.state);

                if state.is_closed() {
                    // Set the flag again so that the waker is released in drop
                    State::set_tx_task(&inner.state);
                    coop.made_progress();
                    return Ready(());
                } else {
                    // SAFETY:
                    // Contract from `Task::drop_task`: initialized `Waker` and *exclusive*
                    // access to the cell.
                    // Evidence:
                    // - Initialization: as argued above, plus `unset_tx_task` returned a
                    //   state whose pre-image had `TX_TASK_SET` set.
                    // - Exclusivity: `unset_tx_task` cleared `TX_TASK_SET` with an `AcqRel`
                    //   `fetch_and`, and the branch condition establishes that `CLOSED` was
                    //   not set at that moment. The receiver reads `tx_task` only from
                    //   `close()`, and only when the state it read still had `TX_TASK_SET`
                    //   set; since our `fetch_and` is a single atomic RMW, any `close()`
                    //   whose `fetch_or` is ordered after it observes the bit as clear and
                    //   will not touch `tx_task`. A `close()` ordered before it would have
                    //   set `CLOSED`, which this branch has ruled out. So no reader remains.
                    // Postcondition: the cell is uninitialized, and `TX_TASK_SET` is
                    // already clear, so the invariant relating the two is restored.
                    unsafe { inner.tx_task.drop_task() };
                }
            }
        }

        if !state.is_tx_task_set() {
            // Attempt to set the task
            // SAFETY:
            // Contract from `Task::set_task`: the cell is uninitialized and this thread has
            // exclusive access to it.
            // Evidence:
            // - Uninitialized: `TX_TASK_SET` is clear, either from the load above or
            //   because the `drop_task` branch just cleared it, and that bit's documented
            //   invariant ties it to initialization of `tx_task`.
            // - Exclusivity: `tx_task` is written only by the sender, and `poll_closed`
            //   holds `&mut self` on the unique `Sender`. A concurrent receiver in
            //   `close()` only reads `tx_task` when it observes `TX_TASK_SET` set, which it
            //   currently is not; the bit is only set again by the `set_tx_task` call
            //   below, whose `AcqRel` `fetch_or` is ordered after this write.
            // Postcondition: the cell holds an initialized `Waker`; `set_tx_task` below
            // publishes that fact.
            unsafe {
                inner.tx_task.set_task(cx);
            }

            // Update the state
            state = State::set_tx_task(&inner.state);

            if state.is_closed() {
                coop.made_progress();
                return Ready(());
            }
        }

        Pending
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.as_ref() {
            inner.complete();
            #[cfg(all(tokio_unstable, feature = "tracing"))]
            self.resource_span.in_scope(|| {
                tracing::trace!(
                target: "runtime::resource::state_update",
                tx_dropped = true,
                tx_dropped.op = "override",
                )
            });
        }
    }
}

impl<T> Receiver<T> {
    /// Prevents the associated [`Sender`] handle from sending a value.
    ///
    /// Any `send` operation which happens after calling `close` is guaranteed
    /// to fail. After calling `close`, [`try_recv`] should be called to
    /// receive a value if one was sent **before** the call to `close`
    /// completed.
    ///
    /// This function is useful to perform a graceful shutdown and ensure that a
    /// value will not be sent into the channel and never received.
    ///
    /// `close` is no-op if a message is already received or the channel
    /// is already closed.
    ///
    /// [`Sender`]: Sender
    /// [`try_recv`]: Receiver::try_recv
    ///
    /// # Examples
    ///
    /// Prevent a value from being sent
    ///
    /// ```
    /// use tokio::sync::oneshot;
    /// use tokio::sync::oneshot::error::TryRecvError;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (tx, mut rx) = oneshot::channel();
    ///
    /// assert!(!tx.is_closed());
    ///
    /// rx.close();
    ///
    /// assert!(tx.is_closed());
    /// assert!(tx.send("never received").is_err());
    ///
    /// match rx.try_recv() {
    ///     Err(TryRecvError::Closed) => {}
    ///     _ => unreachable!(),
    /// }
    /// # }
    /// ```
    ///
    /// Receive a value sent **before** calling `close`
    ///
    /// ```
    /// use tokio::sync::oneshot;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (tx, mut rx) = oneshot::channel();
    ///
    /// assert!(tx.send("will receive").is_ok());
    ///
    /// rx.close();
    ///
    /// let msg = rx.try_recv().unwrap();
    /// assert_eq!(msg, "will receive");
    /// # }
    /// ```
    pub fn close(&mut self) {
        if let Some(inner) = self.inner.as_ref() {
            inner.close();
            #[cfg(all(tokio_unstable, feature = "tracing"))]
            self.resource_span.in_scope(|| {
                tracing::trace!(
                target: "runtime::resource::state_update",
                rx_dropped = true,
                rx_dropped.op = "override",
                )
            });
        }
    }

    /// Checks if this receiver is terminated.
    ///
    /// This function returns true if this receiver has already yielded a [`Poll::Ready`] result.
    /// If so, this receiver should no longer be polled.
    ///
    /// # Examples
    ///
    /// Sending a value and polling it.
    ///
    /// ```
    /// use tokio::sync::oneshot;
    ///
    /// use std::task::Poll;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (tx, mut rx) = oneshot::channel();
    ///
    /// // A receiver is not terminated when it is initialized.
    /// assert!(!rx.is_terminated());
    ///
    /// // A receiver is not terminated it is polled and is still pending.
    /// let poll = futures::poll!(&mut rx);
    /// assert_eq!(poll, Poll::Pending);
    /// assert!(!rx.is_terminated());
    ///
    /// // A receiver is not terminated if a value has been sent, but not yet read.
    /// tx.send(0).unwrap();
    /// assert!(!rx.is_terminated());
    ///
    /// // A receiver *is* terminated after it has been polled and yielded a value.
    /// assert_eq!((&mut rx).await, Ok(0));
    /// assert!(rx.is_terminated());
    /// # }
    /// ```
    ///
    /// Dropping the sender.
    ///
    /// ```
    /// use tokio::sync::oneshot;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (tx, mut rx) = oneshot::channel::<()>();
    ///
    /// // A receiver is not immediately terminated when the sender is dropped.
    /// drop(tx);
    /// assert!(!rx.is_terminated());
    ///
    /// // A receiver *is* terminated after it has been polled and yielded an error.
    /// let _ = (&mut rx).await.unwrap_err();
    /// assert!(rx.is_terminated());
    /// # }
    /// ```
    pub fn is_terminated(&self) -> bool {
        self.inner.is_none()
    }

    /// Checks if a channel is empty.
    ///
    /// This method returns `true` if the channel has no messages.
    ///
    /// It is not necessarily safe to poll an empty receiver, which may have
    /// already yielded a value. Use [`is_terminated()`][Self::is_terminated]
    /// to check whether or not a receiver can be safely polled, instead.
    ///
    /// # Examples
    ///
    /// Sending a value.
    ///
    /// ```
    /// use tokio::sync::oneshot;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (tx, mut rx) = oneshot::channel();
    /// assert!(rx.is_empty());
    ///
    /// tx.send(0).unwrap();
    /// assert!(!rx.is_empty());
    ///
    /// let _ = (&mut rx).await;
    /// assert!(rx.is_empty());
    /// # }
    /// ```
    ///
    /// Dropping the sender.
    ///
    /// ```
    /// use tokio::sync::oneshot;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (tx, mut rx) = oneshot::channel::<()>();
    ///
    /// // A channel is empty if the sender is dropped.
    /// drop(tx);
    /// assert!(rx.is_empty());
    ///
    /// // A closed channel still yields an error, however.
    /// (&mut rx).await.expect_err("should yield an error");
    /// assert!(rx.is_empty());
    /// # }
    /// ```
    ///
    /// Terminated channels are empty.
    ///
    /// ```should_panic,ignore-wasm
    /// use tokio::sync::oneshot;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let (tx, mut rx) = oneshot::channel();
    ///     tx.send(0).unwrap();
    ///     let _ = (&mut rx).await;
    ///
    ///     // NB: an empty channel is not necessarily safe to poll!
    ///     assert!(rx.is_empty());
    ///     let _ = (&mut rx).await;
    /// }
    /// ```
    pub fn is_empty(&self) -> bool {
        let Some(inner) = self.inner.as_ref() else {
            // The channel has already terminated.
            return true;
        };

        let state = State::load(&inner.state, Acquire);
        if state.is_complete() {
            // SAFETY: If `state.is_complete()` returns true, then the
            // `VALUE_SENT` bit has been set and the sender side of the
            // channel will no longer attempt to access the inner
            // `UnsafeCell`. Therefore, it is now safe for us to access the
            // cell.
            //
            // The channel is empty if it does not have a value.
            unsafe { !inner.has_value() }
        } else {
            // The receiver closed the channel or no value has been sent yet.
            true
        }
    }

    /// Attempts to receive a value.
    ///
    /// If a pending value exists in the channel, it is returned. If no value
    /// has been sent, the current task **will not** be registered for
    /// future notification.
    ///
    /// This function is useful to call from outside the context of an
    /// asynchronous task.
    ///
    /// Note that unlike the `poll` method, the `try_recv` method cannot fail
    /// spuriously. Any send or close event that happens before this call to
    /// `try_recv` will be correctly returned to the caller.
    ///
    /// # Return
    ///
    /// - `Ok(T)` if a value is pending in the channel.
    /// - `Err(TryRecvError::Empty)` if no value has been sent yet.
    /// - `Err(TryRecvError::Closed)` if the sender has dropped without sending
    ///   a value, or if the message has already been received.
    ///
    /// # Examples
    ///
    /// `try_recv` before a value is sent, then after.
    ///
    /// ```
    /// use tokio::sync::oneshot;
    /// use tokio::sync::oneshot::error::TryRecvError;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (tx, mut rx) = oneshot::channel();
    ///
    /// match rx.try_recv() {
    ///     // The channel is currently empty
    ///     Err(TryRecvError::Empty) => {}
    ///     _ => unreachable!(),
    /// }
    ///
    /// // Send a value
    /// tx.send("hello").unwrap();
    ///
    /// match rx.try_recv() {
    ///      Ok(value) => assert_eq!(value, "hello"),
    ///      _ => unreachable!(),
    /// }
    /// # }
    /// ```
    ///
    /// `try_recv` when the sender dropped before sending a value
    ///
    /// ```
    /// use tokio::sync::oneshot;
    /// use tokio::sync::oneshot::error::TryRecvError;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let (tx, mut rx) = oneshot::channel::<()>();
    ///
    /// drop(tx);
    ///
    /// match rx.try_recv() {
    ///     // The channel will never receive a value.
    ///     Err(TryRecvError::Closed) => {}
    ///     _ => unreachable!(),
    /// }
    /// # }
    /// ```
    pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
        let result = if let Some(inner) = self.inner.as_ref() {
            let state = State::load(&inner.state, Acquire);

            if state.is_complete() {
                // SAFETY: If `state.is_complete()` returns true, then the
                // `VALUE_SENT` bit has been set and the sender side of the
                // channel will no longer attempt to access the inner
                // `UnsafeCell`. Therefore, it is now safe for us to access the
                // cell.
                match unsafe { inner.consume_value() } {
                    Some(value) => {
                        #[cfg(all(tokio_unstable, feature = "tracing"))]
                        self.resource_span.in_scope(|| {
                            tracing::trace!(
                            target: "runtime::resource::state_update",
                            value_received = true,
                            value_received.op = "override",
                            )
                        });
                        Ok(value)
                    }
                    None => Err(TryRecvError::Closed),
                }
            } else if state.is_closed() {
                Err(TryRecvError::Closed)
            } else {
                // Not ready, this does not clear `inner`
                return Err(TryRecvError::Empty);
            }
        } else {
            Err(TryRecvError::Closed)
        };

        self.inner = None;
        result
    }

    /// Blocking receive to call outside of asynchronous contexts.
    ///
    /// # Panics
    ///
    /// This function panics if called within an asynchronous execution
    /// context.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use std::thread;
    /// use tokio::sync::oneshot;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let (tx, rx) = oneshot::channel::<u8>();
    ///
    ///     let sync_code = thread::spawn(move || {
    ///         assert_eq!(Ok(10), rx.blocking_recv());
    ///     });
    ///
    ///     let _ = tx.send(10);
    ///     sync_code.join().unwrap();
    /// }
    /// # }
    /// ```
    #[track_caller]
    #[cfg(feature = "sync")]
    #[cfg_attr(docsrs, doc(alias = "recv_blocking"))]
    pub fn blocking_recv(self) -> Result<T, RecvError> {
        crate::future::block_on(self)
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.as_ref() {
            let state = inner.close();

            if state.is_complete() {
                // SAFETY: we have ensured that the `VALUE_SENT` bit has been set,
                // so only the receiver can access the value.
                drop(unsafe { inner.consume_value() });
            }

            #[cfg(all(tokio_unstable, feature = "tracing"))]
            self.resource_span.in_scope(|| {
                tracing::trace!(
                target: "runtime::resource::state_update",
                rx_dropped = true,
                rx_dropped.op = "override",
                )
            });
        }
    }
}

impl<T> Future for Receiver<T> {
    type Output = Result<T, RecvError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        #[cfg(all(tokio_unstable, feature = "tracing"))]
        let _res_span = this.resource_span.enter();
        #[cfg(all(tokio_unstable, feature = "tracing"))]
        let _ao_span = this.async_op_span.enter();
        #[cfg(all(tokio_unstable, feature = "tracing"))]
        let _ao_poll_span = this.async_op_poll_span.enter();

        // If `inner` is `None`, then `poll()` has already completed.
        let ret = if let Some(inner) = this.inner.as_ref() {
            #[cfg(all(tokio_unstable, feature = "tracing"))]
            let res = ready!(trace_poll_op!("poll_recv", inner.poll_recv(cx)));

            #[cfg(any(not(tokio_unstable), not(feature = "tracing")))]
            let res = ready!(inner.poll_recv(cx));

            res
        } else {
            panic!("called after complete");
        };

        this.inner = None;
        Ready(ret)
    }
}

impl<T> Inner<T> {
    fn complete(&self) -> bool {
        let prev = State::set_complete(&self.state);

        if prev.is_closed() {
            return false;
        }

        if prev.is_rx_task_set() {
            // TODO: Consume waker?
            // SAFETY:
            // Contract from `Task::with_task`: the cell holds an initialized `Waker` and no
            // concurrent `set_task`/`drop_task` runs on it.
            // Evidence:
            // - Initialization: `prev` is the pre-image of the `AcqRel` CAS in
            //   `set_complete` and has `RX_TASK_SET` set, which by that bit's documented
            //   invariant means `rx_task` is initialized. The `AcqRel` CAS synchronizes
            //   with the receiver's `AcqRel` `fetch_or` in `set_rx_task`, so the `Waker`
            //   stored before it is visible here.
            // - Exclusivity against writers: `rx_task` is written only by the receiver, in
            //   `poll_recv` (`set_task`/`drop_task`) and `close()`/`Inner::drop`. Reaching
            //   this line means our CAS successfully set `VALUE_SENT` while `CLOSED` was
            //   clear. The receiver's `unset_rx_task` + `drop_task` pair in `poll_recv` is
            //   guarded by a re-check of `is_complete()` that takes the "already complete"
            //   branch instead of dropping once `VALUE_SENT` is visible, and `close()`
            //   drops only when `prev` shows `VALUE_SENT` was not yet set — which our CAS
            //   makes impossible for any `close()` ordered after it, while a `close()`
            //   ordered before it would have set `CLOSED` and made our CAS break early.
            //   `Inner::drop` needs `&mut self`, which cannot happen while the sender still
            //   holds a reference.
            unsafe {
                self.rx_task.with_task(Waker::wake_by_ref);
            }
        }

        true
    }

    fn poll_recv(&self, cx: &mut Context<'_>) -> Poll<Result<T, RecvError>> {
        ready!(crate::trace::trace_leaf());
        // Keep track of task budget
        let coop = ready!(crate::task::coop::poll_proceed(cx));

        // Load the state
        let mut state = State::load(&self.state, Acquire);

        if state.is_complete() {
            coop.made_progress();
            // SAFETY:
            // Contract from `consume_value`: only one side may call it at a time; the
            // receiver may call it once `VALUE_SENT` is set.
            // Evidence: the `Acquire` load above observed `VALUE_SENT`. That bit is set
            // only by the `AcqRel` CAS in `set_complete`, which our `Acquire` load
            // synchronizes with, so the sender's write of the value happens-before this
            // read. By the bit's documented invariant the sender performs no further
            // access to the cell once it has set `VALUE_SENT` and seen the CAS succeed,
            // and `poll_recv` is reached only through `&mut Receiver`, so there is no
            // second receiver.
            // Postcondition: the cell is left holding `None`, so a later `consume_value`
            // (e.g. from `Inner::drop`) cannot double-take the value.
            match unsafe { self.consume_value() } {
                Some(value) => Ready(Ok(value)),
                None => Ready(Err(RecvError(()))),
            }
        } else if state.is_closed() {
            coop.made_progress();
            Ready(Err(RecvError(())))
        } else {
            if state.is_rx_task_set() {
                // SAFETY:
                // Contract from `Task::will_wake`: the cell holds an initialized `Waker`
                // and no concurrent `set_task`/`drop_task` runs on it.
                // Evidence:
                // - Initialization: `RX_TASK_SET` is set in the `Acquire` load above,
                //   which by that bit's documented invariant means `rx_task` is
                //   initialized; the load synchronizes with the `AcqRel` `fetch_or` in
                //   `set_rx_task` that published it.
                // - Exclusivity against writers: `rx_task` is written only by the
                //   receiver, and `poll_recv` is reached only through `&mut Receiver`, so
                //   this is the only receiver. The sender only ever *reads* `rx_task`
                //   (`with_task` in `complete()`), which is a permitted concurrent shared
                //   access.
                let will_notify = unsafe { self.rx_task.will_wake(cx) };

                // Check if the task is still the same
                if !will_notify {
                    // Unset the task
                    state = State::unset_rx_task(&self.state);
                    if state.is_complete() {
                        // Set the flag again so that the waker is released in drop
                        State::set_rx_task(&self.state);

                        coop.made_progress();
                        // SAFETY: If `state.is_complete()` returns true, then the
                        // `VALUE_SENT` bit has been set and the sender side of the
                        // channel will no longer attempt to access the inner
                        // `UnsafeCell`. Therefore, it is now safe for us to access the
                        // cell.
                        return match unsafe { self.consume_value() } {
                            Some(value) => Ready(Ok(value)),
                            None => Ready(Err(RecvError(()))),
                        };
                    } else {
                        // SAFETY:
                        // Contract from `Task::drop_task`: initialized `Waker` and
                        // *exclusive* access to the cell.
                        // Evidence:
                        // - Initialization: as argued for `will_wake` above;
                        //   `unset_rx_task` returned a state whose pre-image had
                        //   `RX_TASK_SET` set.
                        // - Exclusivity: `unset_rx_task` cleared `RX_TASK_SET` with an
                        //   `AcqRel` `fetch_and`, and this branch establishes that
                        //   `VALUE_SENT` was still clear at that point. The sender reads
                        //   `rx_task` only from `complete()`, and only after its CAS has
                        //   set `VALUE_SENT` *and* it observed `RX_TASK_SET` in that CAS's
                        //   pre-image. Because both are single atomic RMWs on the same
                        //   location, they are totally ordered: a `complete()` ordered
                        //   after our `fetch_and` sees `RX_TASK_SET` clear and does not
                        //   read the cell, and one ordered before it would have made
                        //   `VALUE_SENT` visible in the state we just read, taking the
                        //   `is_complete()` branch above instead. So no reader remains.
                        // Postcondition: the cell is uninitialized and `RX_TASK_SET` is
                        // already clear, so the invariant relating the two is restored.
                        unsafe { self.rx_task.drop_task() };
                    }
                }
            }

            if !state.is_rx_task_set() {
                // Attempt to set the task
                // SAFETY:
                // Contract from `Task::set_task`: the cell is uninitialized and this
                // thread has exclusive access to it.
                // Evidence:
                // - Uninitialized: `RX_TASK_SET` is clear, either from the load above or
                //   because the `drop_task` branch just cleared it, and that bit's
                //   documented invariant ties it to initialization of `rx_task`.
                // - Exclusivity: `rx_task` is written only by the receiver, and this is
                //   the only receiver (`&mut Receiver`). A concurrent sender in
                //   `complete()` reads `rx_task` only when its CAS pre-image shows
                //   `RX_TASK_SET` set, which it currently is not; the bit is only set
                //   again by the `set_rx_task` call below, whose `AcqRel` `fetch_or` is
                //   ordered after this write.
                // Postcondition: the cell holds an initialized `Waker`; `set_rx_task`
                // below publishes that fact so the eventual `drop_task` runs.
                unsafe {
                    self.rx_task.set_task(cx);
                }

                // Update the state
                state = State::set_rx_task(&self.state);

                if state.is_complete() {
                    coop.made_progress();
                    // SAFETY:
                    // Contract from `consume_value`: the receiver may call it once
                    // `VALUE_SENT` is set.
                    // Evidence: `set_rx_task`'s `AcqRel` `fetch_or` returned a state with
                    // `VALUE_SENT` set, and that RMW synchronizes with the `AcqRel` CAS in
                    // `set_complete` that set it, so the sender's write of the value
                    // happens-before this read. Per that bit's documented invariant the
                    // sender performs no further access to the cell, and this is the only
                    // receiver.
                    // Postcondition: the cell is left holding `None`.
                    match unsafe { self.consume_value() } {
                        Some(value) => Ready(Ok(value)),
                        None => Ready(Err(RecvError(()))),
                    }
                } else {
                    Pending
                }
            } else {
                Pending
            }
        }
    }

    /// Called by `Receiver` to indicate that the value will never be received.
    fn close(&self) -> State {
        let prev = State::set_closed(&self.state);

        if prev.is_tx_task_set() && !prev.is_complete() {
            // SAFETY:
            // Contract from `Task::with_task`: the cell holds an initialized `Waker` and
            // no concurrent `set_task`/`drop_task` runs on it.
            // Evidence:
            // - Initialization: `prev` is the pre-image of the `Acquire` `fetch_or` in
            //   `set_closed` and has `TX_TASK_SET` set, which by that bit's documented
            //   invariant means `tx_task` is initialized. That RMW reads the value written
            //   by the sender's `AcqRel` `fetch_or` in `set_tx_task`, so the `Waker` is
            //   visible here.
            // - Exclusivity against writers: the sender writes `tx_task` only in
            //   `poll_closed`, and only along paths guarded by the `CLOSED` bit. Our
            //   `fetch_or` set `CLOSED` in a single atomic RMW; a `poll_closed` ordered
            //   after it observes `CLOSED` and returns `Ready(())` before touching the
            //   cell (it re-checks `is_closed()` after `unset_tx_task` and re-sets the bit
            //   rather than dropping). `Sender::drop` also only drops the waker via
            //   `Inner::drop`, which requires `&mut Inner` and so cannot run while we hold
            //   a shared reference.
            unsafe {
                self.tx_task.with_task(Waker::wake_by_ref);
            }
        }

        if prev.is_rx_task_set() && !prev.is_complete() {
            State::unset_rx_task(&self.state);
            // SAFETY: The sender only accesses `rx_task` (via
            // `wake_by_ref`) in `complete()` after successfully setting
            // `VALUE_SENT`. But `set_complete` will not set `VALUE_SENT`
            // if `CLOSED` is already set (its CAS loop breaks early).
            // Since `prev` shows that `VALUE_SENT` was not set before we
            // set `CLOSED`, the sender can no longer set `VALUE_SENT` and
            // will never access `rx_task`. Therefore, we have exclusive
            // access here.
            unsafe { self.rx_task.drop_task() };
        }

        prev
    }

    /// Consumes the value. This function does not check `state`.
    ///
    /// # Safety
    ///
    /// The caller must have exclusive access to the `value` cell for the duration of this
    /// call — no other thread may be reading from or writing to it. The `VALUE_SENT` state
    /// bit is what partitions that access between the two sides: if `VALUE_SENT` is not
    /// set, then only the sender may call this method; if it is set, then only the
    /// receiver may. The caller must also establish the corresponding happens-before edge
    /// through `state` (an `Acquire` read that synchronizes with the other side's release)
    /// so that the value written by the sender is visible.
    ///
    /// # Postcondition
    ///
    /// The cell is left holding `None`, and ownership of any value it held is transferred
    /// to the caller. A second `consume_value` therefore yields `None` rather than
    /// duplicating the value.
    unsafe fn consume_value(&self) -> Option<T> {
        // SAFETY:
        // Operation: `(*ptr).take()` through the `value` cell — a read-modify-write of the
        // `Option<T>` it holds.
        // Required contract: `ptr` must be non-null, aligned, and valid for reads and
        // writes of `Option<T>`, pointing to an initialized `Option<T>`, with no
        // concurrent access.
        // Evidence: `UnsafeCell::with_mut` yields a non-null, aligned pointer to the
        // cell's contents; the cell is live because `&self` borrows the `Inner`; the
        // `Option<T>` is initialized by `Inner::new` and only ever assigned whole values.
        // Exclusivity and visibility are this function's `# Safety` precondition, which
        // every caller discharges from the `VALUE_SENT` bit.
        self.value.with_mut(|ptr| unsafe { (*ptr).take() })
    }

    /// Returns true if there is a value. This function does not check `state`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other thread is concurrently *writing* to the
    /// `value` cell for the duration of this call. As for [`Self::consume_value`], the
    /// `VALUE_SENT` state bit partitions that access: if `VALUE_SENT` is not set, then
    /// only the sender may call this method; if it is set, then only the receiver may.
    unsafe fn has_value(&self) -> bool {
        // SAFETY:
        // Operation: `(*ptr).is_some()` through the `value` cell — a shared read of the
        // `Option<T>` it holds.
        // Required contract: `ptr` must be non-null, aligned, valid for reads of
        // `Option<T>`, point to an initialized `Option<T>`, and not be written
        // concurrently.
        // Evidence: `UnsafeCell::with` yields a non-null, aligned pointer to the cell's
        // contents; the cell is live because `&self` borrows the `Inner`; the `Option<T>`
        // is initialized by `Inner::new`. Absence of a concurrent writer is this
        // function's `# Safety` precondition.
        self.value.with(|ptr| unsafe { (*ptr).is_some() })
    }
}

// SAFETY: Implementer obligation of `Send`: transferring an `Inner<T>` to another thread
// transfers the `Option<T>` in its `value` cell and the `Waker`s in `rx_task`/`tx_task`.
// `T: Send` licenses moving the payload; `Waker` is unconditionally `Send + Sync`, and
// `AtomicUsize` is too. No shared access to `T` is created by sending, so `T: Sync` is not
// required.
unsafe impl<T: Send> Send for Inner<T> {}
// SAFETY: Implementer obligation of `Sync`: `&Inner<T>` must be usable from the sender
// thread and the receiver thread at once. `Inner<T>` contains `UnsafeCell`s, which are
// never `Sync`, so this impl must be written by hand and its soundness rests on the
// `state` protocol rather than on the field types:
//
// - `value` is accessed only through `consume_value`/`has_value`, whose `# Safety`
//   contract restricts access to exactly one side at a time, selected by the `VALUE_SENT`
//   bit. Every caller discharges that from an `Acquire` load of, or an `AcqRel` RMW on,
//   `state`, which synchronizes with the `AcqRel` CAS in `set_complete`. So the two sides'
//   accesses to `value` are ordered, never concurrent.
// - `rx_task` and `tx_task` are each written by only one side, and read by the other only
//   while the corresponding `*_TASK_SET` bit is observed set in an atomic RMW on `state`;
//   the per-call proofs above show the writer cannot be running concurrently with such a
//   reader.
//
// `T` never crosses as a shared reference — the receiver *moves* the value out — so the
// requirement is `T: Send`, not `T: Sync`. This is the same reasoning as `Mutex<T>: Sync`
// where `T: Send`.
unsafe impl<T: Send> Sync for Inner<T> {}

fn mut_load(this: &mut AtomicUsize) -> usize {
    this.with_mut(|v| *v)
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        let state = State(mut_load(&mut self.state));

        // SAFETY (both blocks below):
        // Contract from `Task::drop_task`: initialized `Waker` and exclusive access.
        // Evidence:
        // - Exclusivity: this is `Drop::drop`, so we hold `&mut self`. Both the `Sender`
        //   and the `Receiver` have released their handles to this `Inner`, so no other
        //   thread can reach either cell.
        // - Initialization: `state` was read with `mut_load`, i.e. through `&mut`, so it
        //   is the final value; the `*_TASK_SET` bits it reports are authoritative and, by
        //   those bits' documented invariants, each set bit means the corresponding cell
        //   holds an initialized `Waker`.
        // Postcondition: each `Waker` is dropped exactly once. The state bits are not
        // cleared afterwards, which is fine because `self` is being destroyed and nobody
        // can observe them again.
        if state.is_rx_task_set() {
            // SAFETY: see the shared proof directly above.
            unsafe {
                self.rx_task.drop_task();
            }
        }

        if state.is_tx_task_set() {
            // SAFETY: see the shared proof directly above.
            unsafe {
                self.tx_task.drop_task();
            }
        }

        // SAFETY: we have `&mut self`, and therefore we have
        // exclusive access to the value.
        unsafe {
            // Note: the assertion holds because if the value has been sent by sender,
            // we must ensure that the value must have been consumed by the receiver before
            // dropping the `Inner`.
            debug_assert!(self.consume_value().is_none());
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Inner<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        use std::sync::atomic::Ordering::Relaxed;

        fmt.debug_struct("Inner")
            .field("state", &State::load(&self.state, Relaxed))
            .finish()
    }
}

/// Indicates that a waker for the receiving task has been set.
///
/// # Safety
///
/// If this bit is not set, the `rx_task` field may be uninitialized.
const RX_TASK_SET: usize = 0b00001;
/// Indicates that a value has been stored in the channel's inner `UnsafeCell`.
///
/// # Safety
///
/// This bit controls which side of the channel is permitted to access the
/// `UnsafeCell`. If it is set, the `UnsafeCell` may ONLY be accessed by the
/// receiver. If this bit is NOT set, the `UnsafeCell` may ONLY be accessed by
/// the sender.
const VALUE_SENT: usize = 0b00010;
const CLOSED: usize = 0b00100;

/// Indicates that a waker for the sending task has been set.
///
/// # Safety
///
/// If this bit is not set, the `tx_task` field may be uninitialized.
const TX_TASK_SET: usize = 0b01000;

impl State {
    fn new() -> State {
        State(0)
    }

    fn is_complete(self) -> bool {
        self.0 & VALUE_SENT == VALUE_SENT
    }

    fn set_complete(cell: &AtomicUsize) -> State {
        // This method is a compare-and-swap loop rather than a fetch-or like
        // other `set_$WHATEVER` methods on `State`. This is because we must
        // check if the state has been closed before setting the `VALUE_SENT`
        // bit.
        //
        // We don't want to set both the `VALUE_SENT` bit if the `CLOSED`
        // bit is already set, because `VALUE_SENT` will tell the receiver that
        // it's okay to access the inner `UnsafeCell`. Immediately after calling
        // `set_complete`, if the channel was closed, the sender will _also_
        // access the `UnsafeCell` to take the value back out, so if a
        // `poll_recv` or `try_recv` call is occurring concurrently, both
        // threads may try to access the `UnsafeCell` if we were to set the
        // `VALUE_SENT` bit on a closed channel.
        let mut state = cell.load(Ordering::Relaxed);
        loop {
            if State(state).is_closed() {
                break;
            }
            // TODO: This could be `Release`, followed by an `Acquire` fence *if*
            // the `RX_TASK_SET` flag is set. However, `loom` does not support
            // fences yet.
            match cell.compare_exchange_weak(
                state,
                state | VALUE_SENT,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => state = actual,
            }
        }
        State(state)
    }

    fn is_rx_task_set(self) -> bool {
        self.0 & RX_TASK_SET == RX_TASK_SET
    }

    fn set_rx_task(cell: &AtomicUsize) -> State {
        let val = cell.fetch_or(RX_TASK_SET, AcqRel);
        State(val | RX_TASK_SET)
    }

    fn unset_rx_task(cell: &AtomicUsize) -> State {
        let val = cell.fetch_and(!RX_TASK_SET, AcqRel);
        State(val & !RX_TASK_SET)
    }

    fn is_closed(self) -> bool {
        self.0 & CLOSED == CLOSED
    }

    fn set_closed(cell: &AtomicUsize) -> State {
        // Acquire because we want all later writes (attempting to poll) to be
        // ordered after this.
        let val = cell.fetch_or(CLOSED, Acquire);
        State(val)
    }

    fn set_tx_task(cell: &AtomicUsize) -> State {
        let val = cell.fetch_or(TX_TASK_SET, AcqRel);
        State(val | TX_TASK_SET)
    }

    fn unset_tx_task(cell: &AtomicUsize) -> State {
        let val = cell.fetch_and(!TX_TASK_SET, AcqRel);
        State(val & !TX_TASK_SET)
    }

    fn is_tx_task_set(self) -> bool {
        self.0 & TX_TASK_SET == TX_TASK_SET
    }

    fn as_usize(self) -> usize {
        self.0
    }

    fn load(cell: &AtomicUsize, order: Ordering) -> State {
        let val = cell.load(order);
        State(val)
    }
}

impl fmt::Debug for State {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("State")
            .field("is_complete", &self.is_complete())
            .field("is_closed", &self.is_closed())
            .field("is_rx_task_set", &self.is_rx_task_set())
            .field("is_tx_task_set", &self.is_tx_task_set())
            .finish()
    }
}
