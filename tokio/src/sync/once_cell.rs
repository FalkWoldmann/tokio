use super::{Semaphore, SemaphorePermit, TryAcquireError};
use crate::loom::cell::UnsafeCell;
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::mem::MaybeUninit;
use std::ops::Drop;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

// This file contains an implementation of an OnceCell. The principle
// behind the safety of the cell is that any thread with an `&OnceCell` may
// access the `value` field according the following rules:
//
//  1. When `value_set` is false, the `value` field may be modified by the
//     thread holding the permit on the semaphore.
//  2. When `value_set` is true, the `value` field may be accessed immutably by
//     any thread.
//
// It is an invariant that if the semaphore is closed, then `value_set` is true.
// The reverse does not necessarily hold — but if not, the semaphore may not
// have any available permits.
//
// A thread with a `&mut OnceCell` may modify the value in any way it wants as
// long as the invariants are upheld.

/// A thread-safe cell that can be written to only once.
///
/// A `OnceCell` is typically used for global variables that need to be
/// initialized once on first use, but need no further changes. The `OnceCell`
/// in Tokio allows the initialization procedure to be asynchronous.
///
/// # Examples
///
/// ```
/// use tokio::sync::OnceCell;
///
/// async fn some_computation() -> u32 {
///     1 + 1
/// }
///
/// static ONCE: OnceCell<u32> = OnceCell::const_new();
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let result = ONCE.get_or_init(some_computation).await;
/// assert_eq!(*result, 2);
/// # }
/// ```
///
/// It is often useful to write a wrapper method for accessing the value.
///
/// ```
/// use tokio::sync::OnceCell;
///
/// static ONCE: OnceCell<u32> = OnceCell::const_new();
///
/// async fn get_global_integer() -> &'static u32 {
///     ONCE.get_or_init(|| async {
///         1 + 1
///     }).await
/// }
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let result = get_global_integer().await;
/// assert_eq!(*result, 2);
/// # }
/// ```
pub struct OnceCell<T> {
    value_set: AtomicBool,
    value: UnsafeCell<MaybeUninit<T>>,
    semaphore: Semaphore,
}

impl<T> Default for OnceCell<T> {
    fn default() -> OnceCell<T> {
        OnceCell::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for OnceCell<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("OnceCell")
            .field("value", &self.get())
            .finish()
    }
}

impl<T: Clone> Clone for OnceCell<T> {
    fn clone(&self) -> OnceCell<T> {
        OnceCell::new_with(self.get().cloned())
    }
}

impl<T: PartialEq> PartialEq for OnceCell<T> {
    fn eq(&self, other: &OnceCell<T>) -> bool {
        self.get() == other.get()
    }
}

impl<T: Eq> Eq for OnceCell<T> {}

impl<T> Drop for OnceCell<T> {
    fn drop(&mut self) {
        if self.initialized_mut() {
            // SAFETY:
            // Justifies two operations.
            // 1. `(*ptr).as_mut_ptr()` — `UnsafeCell::with_mut` yields a non-null, aligned
            //    pointer to the `MaybeUninit<T>`, and `MaybeUninit::as_mut_ptr` requires
            //    only a valid place, not an initialized one.
            // 2. `ptr::drop_in_place(..)` — contract: the pointer must be non-null,
            //    properly aligned, valid for reads and writes, and point to a valid
            //    (initialized) `T` that nothing else accesses while the destructor runs.
            //    Evidence: non-null and aligned from (1); initialization from
            //    `initialized_mut()`, which reads `value_set` through `get_mut` — a
            //    non-atomic read that is exact because we hold `&mut self`. By the cell's
            //    protocol `value_set` is set to `true` only by `set_value`, after it has
            //    written an owned `T`. Exclusivity holds because this is `Drop::drop`, so
            //    we have `&mut self` and no other reference to the cell can exist.
            // Postcondition: the `T` is dropped exactly once. `value_set` is not reset,
            // which is fine because the cell is being destroyed; `into_inner` *does* reset
            // it precisely so that this destructor does not run a second time.
            unsafe {
                self.value
                    .with_mut(|ptr| ptr::drop_in_place((*ptr).as_mut_ptr()));
            };
        }
    }
}

impl<T> From<T> for OnceCell<T> {
    fn from(value: T) -> Self {
        OnceCell {
            value_set: AtomicBool::new(true),
            value: UnsafeCell::new(MaybeUninit::new(value)),
            semaphore: Semaphore::new_closed(),
        }
    }
}

impl<T> OnceCell<T> {
    /// Creates a new empty `OnceCell` instance.
    pub fn new() -> Self {
        OnceCell {
            value_set: AtomicBool::new(false),
            value: UnsafeCell::new(MaybeUninit::uninit()),
            semaphore: Semaphore::new(1),
        }
    }

    /// Creates a new empty `OnceCell` instance.
    ///
    /// Equivalent to `OnceCell::new`, except that it can be used in static
    /// variables.
    ///
    /// When using the `tracing` [unstable feature], a `OnceCell` created with
    /// `const_new` will not be instrumented. As such, it will not be visible
    /// in [`tokio-console`]. Instead, [`OnceCell::new`] should be used to
    /// create an instrumented object if that is needed.
    ///
    /// # Example
    ///
    /// ```
    /// use tokio::sync::OnceCell;
    ///
    /// static ONCE: OnceCell<u32> = OnceCell::const_new();
    ///
    /// async fn get_global_integer() -> &'static u32 {
    ///     ONCE.get_or_init(|| async {
    ///         1 + 1
    ///     }).await
    /// }
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let result = get_global_integer().await;
    /// assert_eq!(*result, 2);
    /// # }
    /// ```
    ///
    /// [`tokio-console`]: https://github.com/tokio-rs/console
    /// [unstable feature]: crate#unstable-features
    #[cfg(not(all(loom, test)))]
    pub const fn const_new() -> Self {
        OnceCell {
            value_set: AtomicBool::new(false),
            value: UnsafeCell::new(MaybeUninit::uninit()),
            semaphore: Semaphore::const_new(1),
        }
    }

    /// Creates a new `OnceCell` that contains the provided value, if any.
    ///
    /// If the `Option` is `None`, this is equivalent to `OnceCell::new`.
    ///
    /// [`OnceCell::new`]: crate::sync::OnceCell::new
    // Once https://github.com/rust-lang/rust/issues/73255 lands
    // and tokio MSRV is bumped to the rustc version with it stabilised,
    // we can make this function available in const context,
    // by creating `Semaphore::const_new_closed`.
    pub fn new_with(value: Option<T>) -> Self {
        if let Some(v) = value {
            OnceCell::from(v)
        } else {
            OnceCell::new()
        }
    }

    /// Creates a new `OnceCell` that contains the provided value.
    ///
    /// # Example
    ///
    /// When using the `tracing` [unstable feature], a `OnceCell` created with
    /// `const_new_with` will not be instrumented. As such, it will not be
    /// visible in [`tokio-console`]. Instead, [`OnceCell::new_with`] should be
    /// used to create an instrumented object if that is needed.
    ///
    /// ```
    /// use tokio::sync::OnceCell;
    ///
    /// static ONCE: OnceCell<u32> = OnceCell::const_new_with(1);
    ///
    /// async fn get_global_integer() -> &'static u32 {
    ///     ONCE.get_or_init(|| async {
    ///         1 + 1
    ///     }).await
    /// }
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() {
    /// let result = get_global_integer().await;
    /// assert_eq!(*result, 1);
    /// # }
    /// ```
    ///
    /// [`tokio-console`]: https://github.com/tokio-rs/console
    /// [unstable feature]: crate#unstable-features
    #[cfg(not(all(loom, test)))]
    pub const fn const_new_with(value: T) -> Self {
        OnceCell {
            value_set: AtomicBool::new(true),
            value: UnsafeCell::new(MaybeUninit::new(value)),
            semaphore: Semaphore::const_new_closed(),
        }
    }

    /// Returns `true` if the `OnceCell` currently contains a value, and `false`
    /// otherwise.
    pub fn initialized(&self) -> bool {
        // Using acquire ordering so any threads that read a true from this
        // atomic is able to read the value.
        self.value_set.load(Ordering::Acquire)
    }

    /// Returns `true` if the `OnceCell` currently contains a value, and `false`
    /// otherwise.
    fn initialized_mut(&mut self) -> bool {
        *self.value_set.get_mut()
    }

    /// # Safety
    ///
    /// The caller must ensure that the `OnceCell` is initialized — in practice, that they
    /// have observed `value_set` as `true` through an `Acquire` load (or equivalent), so
    /// that the write performed by `set_value` happens-before this read. The value must
    /// also not be moved out or dropped for the returned lifetime, which is guaranteed for
    /// a `OnceCell` because the only paths that do so (`into_inner` and `Drop`) require
    /// ownership or `&mut self`.
    unsafe fn get_unchecked(&self) -> &T {
        // SAFETY:
        // Justifies two operations.
        // 1. `(*ptr).as_ptr()` — `UnsafeCell::with` yields a non-null, aligned pointer to
        //    the `MaybeUninit<T>`, and `MaybeUninit::as_ptr` requires only a valid place.
        // 2. `&*..` — creating `&T`. Contract: non-null, aligned, pointing to an
        //    initialized, valid `T` for the returned lifetime, with no `&mut T` alias.
        //    Evidence: non-null and aligned from (1); initialization is this function's
        //    `# Safety` precondition. No `&mut T` can alias it: `get_unchecked_mut` and
        //    `Drop` both require `&mut self`, which the `&self` here excludes, and
        //    `set_value` writes only while `value_set` is still `false`, which the
        //    precondition rules out. The returned lifetime is tied to `&self`.
        unsafe { &*self.value.with(|ptr| (*ptr).as_ptr()) }
    }

    /// # Safety
    ///
    /// The caller must ensure that the `OnceCell` is initialized. `&mut self` supplies the
    /// exclusivity, so unlike [`Self::get_unchecked`] no synchronisation argument is
    /// needed: the caller can read `value_set` non-atomically via `get_mut`.
    unsafe fn get_unchecked_mut(&mut self) -> &mut T {
        // SAFETY:
        //
        // 1. The caller guarantees that the OnceCell is initialized.
        // 2. The `&mut self` guarantees that there are no other references to the value.
        unsafe { &mut *self.value.with_mut(|ptr| (*ptr).as_mut_ptr()) }
    }

    fn set_value(&self, value: T, permit: SemaphorePermit<'_>) -> &T {
        // SAFETY:
        // Justifies two operations.
        // 1. `(*ptr).as_mut_ptr()` — `UnsafeCell::with_mut` yields a non-null, aligned
        //    pointer to the `MaybeUninit<T>`; `MaybeUninit::as_mut_ptr` needs only a valid
        //    place, not an initialized one.
        // 2. `.write(value)` — contract from `ptr::write`: non-null, aligned, valid for
        //    writes; it does not require the destination to be initialized and does not
        //    drop the old contents.
        //    Evidence: non-null and aligned from (1). Exclusivity: the caller holds
        //    `permit`, and the `OnceCell`'s semaphore is created with a single permit
        //    which is never released — `set_value` calls `semaphore.close()` and
        //    `permit.forget()` below — so exactly one call to `set_value` can ever reach
        //    this line for a given cell. Readers are excluded because they all go through
        //    `initialized()`/`get_unchecked`, which require `value_set == true`, and that
        //    flag is only stored (with `Release`) *after* this write.
        //    Not dropping the old contents is correct rather than a leak: the cell is
        //    uninitialized until this point.
        // Postcondition: the cell owns `value`. The `Release` store below publishes it,
        // pairing with the `Acquire` load in `initialized()`.
        unsafe {
            self.value.with_mut(|ptr| (*ptr).as_mut_ptr().write(value));
        }

        // Using release ordering so any threads that read a true from this
        // atomic is able to read the value we just stored.
        self.value_set.store(true, Ordering::Release);
        self.semaphore.close();
        permit.forget();

        // SAFETY: We just initialized the cell.
        unsafe { self.get_unchecked() }
    }

    /// Returns a reference to the value currently stored in the `OnceCell`, or
    /// `None` if the `OnceCell` is empty.
    pub fn get(&self) -> Option<&T> {
        if self.initialized() {
            // SAFETY:
            // Contract from `get_unchecked`: the cell is initialized, established through
            // an `Acquire` load.
            // Evidence: `initialized()` returned true, and it performs exactly that
            // `Acquire` load of `value_set`, which synchronizes with the `Release` store
            // in `set_value` and therefore orders the value's initialization before this
            // read. `value_set` is never reset while a shared reference can exist
            // (`into_inner` takes `self` by value and `take` takes `&mut self`).
            Some(unsafe { self.get_unchecked() })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the value currently stored in the
    /// `OnceCell`, or `None` if the `OnceCell` is empty.
    ///
    /// Since this call borrows the `OnceCell` mutably, it is safe to mutate the
    /// value inside the `OnceCell` — the mutable borrow statically guarantees
    /// no other references exist.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.initialized_mut() {
            // SAFETY:
            // Contract from `get_unchecked_mut`: the cell is initialized.
            // Evidence: `initialized_mut()` returned true. It reads `value_set` through
            // `AtomicBool::get_mut`, which is sound and exact here because we hold
            // `&mut self`; no synchronisation is needed for the same reason.
            Some(unsafe { self.get_unchecked_mut() })
        } else {
            None
        }
    }

    /// Sets the value of the `OnceCell` to the given value if the `OnceCell` is
    /// empty.
    ///
    /// If the `OnceCell` already has a value, this call will fail with an
    /// [`SetError::AlreadyInitializedError`].
    ///
    /// If the `OnceCell` is empty, but some other task is currently trying to
    /// set the value, this call will fail with [`SetError::InitializingError`].
    ///
    /// [`SetError::AlreadyInitializedError`]: crate::sync::SetError::AlreadyInitializedError
    /// [`SetError::InitializingError`]: crate::sync::SetError::InitializingError
    pub fn set(&self, value: T) -> Result<(), SetError<T>> {
        if self.initialized() {
            return Err(SetError::AlreadyInitializedError(value));
        }

        // Another task might be initializing the cell, in which case
        // `try_acquire` will return an error. If we succeed to acquire the
        // permit, then we can set the value.
        match self.semaphore.try_acquire() {
            Ok(permit) => {
                debug_assert!(!self.initialized());
                self.set_value(value, permit);
                Ok(())
            }
            Err(TryAcquireError::NoPermits) => {
                // Some other task is holding the permit. That task is
                // currently trying to initialize the value.
                Err(SetError::InitializingError(value))
            }
            Err(TryAcquireError::Closed) => {
                // The semaphore was closed. Some other task has initialized
                // the value.
                Err(SetError::AlreadyInitializedError(value))
            }
        }
    }

    /// Gets the value currently in the `OnceCell`, or initialize it with the
    /// given asynchronous operation.
    ///
    /// If some other task is currently working on initializing the `OnceCell`,
    /// this call will wait for that other task to finish, then return the value
    /// that the other task produced.
    ///
    /// If the provided operation is cancelled or panics, the initialization
    /// attempt is cancelled. If there are other tasks waiting for the value to
    /// be initialized, one of them will start another attempt at initializing
    /// the value.
    ///
    /// This will deadlock if `f` tries to initialize the cell recursively.
    pub async fn get_or_init<F, Fut>(&self, f: F) -> &T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        crate::trace::async_trace_leaf().await;

        if self.initialized() {
            // SAFETY: The OnceCell has been fully initialized.
            unsafe { self.get_unchecked() }
        } else {
            // Here we try to acquire the semaphore permit. Holding the permit
            // will allow us to set the value of the OnceCell, and prevents
            // other tasks from initializing the OnceCell while we are holding
            // it.
            match self.semaphore.acquire().await {
                Ok(permit) => {
                    debug_assert!(!self.initialized());

                    // If `f()` panics or `select!` is called, this
                    // `get_or_init` call is aborted and the semaphore permit is
                    // dropped.
                    let value = f().await;

                    self.set_value(value, permit)
                }
                Err(_) => {
                    debug_assert!(self.initialized());

                    // SAFETY: The semaphore has been closed. This only happens
                    // when the OnceCell is fully initialized.
                    unsafe { self.get_unchecked() }
                }
            }
        }
    }

    /// Gets the value currently in the `OnceCell`, or initialize it with the
    /// given asynchronous operation.
    ///
    /// If some other task is currently working on initializing the `OnceCell`,
    /// this call will wait for that other task to finish, then return the value
    /// that the other task produced.
    ///
    /// If the provided operation returns an error, is cancelled or panics, the
    /// initialization attempt is cancelled. If there are other tasks waiting
    /// for the value to be initialized, one of them will start another attempt
    /// at initializing the value.
    ///
    /// This will deadlock if `f` tries to initialize the cell recursively.
    pub async fn get_or_try_init<E, F, Fut>(&self, f: F) -> Result<&T, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        crate::trace::async_trace_leaf().await;

        if self.initialized() {
            // SAFETY: The OnceCell has been fully initialized.
            unsafe { Ok(self.get_unchecked()) }
        } else {
            // Here we try to acquire the semaphore permit. Holding the permit
            // will allow us to set the value of the OnceCell, and prevents
            // other tasks from initializing the OnceCell while we are holding
            // it.
            match self.semaphore.acquire().await {
                Ok(permit) => {
                    debug_assert!(!self.initialized());

                    // If `f()` panics or `select!` is called, this
                    // `get_or_try_init` call is aborted and the semaphore
                    // permit is dropped.
                    let value = f().await;

                    match value {
                        Ok(value) => Ok(self.set_value(value, permit)),
                        Err(e) => Err(e),
                    }
                }
                Err(_) => {
                    debug_assert!(self.initialized());

                    // SAFETY: The semaphore has been closed. This only happens
                    // when the OnceCell is fully initialized.
                    unsafe { Ok(self.get_unchecked()) }
                }
            }
        }
    }

    /// Takes the value from the cell, destroying the cell in the process.
    /// Returns `None` if the cell is empty.
    pub fn into_inner(mut self) -> Option<T> {
        if self.initialized_mut() {
            // Set to uninitialized for the destructor of `OnceCell` to work properly
            *self.value_set.get_mut() = false;
            // SAFETY:
            // Justifies two operations.
            // 1. `ptr::read(ptr)` where `ptr: *const MaybeUninit<T>` — contract: non-null,
            //    aligned, valid for reads, pointing to a valid value at that type.
            //    Evidence: `UnsafeCell::with` yields a non-null, aligned pointer to the
            //    cell's contents; `MaybeUninit<T>` is a valid value for any byte pattern
            //    (std docs), so no initialization premise is needed for this step.
            // 2. `.assume_init()` — contract: the `MaybeUninit<T>` must hold a valid,
            //    initialized `T`.
            //    Evidence: `initialized_mut()` returned true just above, read through
            //    `&mut self` so it is exact, and `value_set` is set to `true` only by
            //    `set_value` after it wrote an owned `T`. Ownership: `into_inner` takes
            //    `self` by value, so no other reference to the cell exists.
            // Postcondition: ownership of the `T` moves to the caller by a bitwise copy,
            // and the cell's bytes still hold that copy. The `value_set = false` store
            // above is what makes this sound: without it, `OnceCell::drop` (which runs on
            // `self` at the end of this function) would drop the same `T` a second time.
            Some(unsafe { self.value.with(|ptr| ptr::read(ptr).assume_init()) })
        } else {
            None
        }
    }

    /// Takes ownership of the current value, leaving the cell empty.  Returns
    /// `None` if the cell is empty.
    pub fn take(&mut self) -> Option<T> {
        std::mem::take(self).into_inner()
    }
}

// SAFETY: Implementer obligation of `Sync`: `&OnceCell<T>` must be usable from several
// threads at once. `OnceCell<T>` holds an `UnsafeCell`, which is never `Sync`, so this
// impl is explicit and rests on the initialisation protocol rather than the field types:
//
// - `get`/`get_or_init` hand out `&T` to any number of threads concurrently once the cell
//   is initialized. Aliased shared access to `T` across threads is exactly `T: Sync`.
// - `set`/`get_or_init` on a `&OnceCell<T>` let a thread deposit a `T` that another thread
//   then observes — a cross-thread ownership transfer — so `T: Send` is also required.
//
// Data-race freedom comes from the single-permit `Semaphore` (only one initialiser can
// ever run `set_value`) plus the `Release` store / `Acquire` load pair on `value_set`,
// which orders the initializing write before every reader's access.
unsafe impl<T: Sync + Send> Sync for OnceCell<T> {}

// SAFETY: Implementer obligation of `Send`: transferring ownership of a `OnceCell<T>` to
// another thread transfers the `T` it may hold, which is exactly what `T: Send` licenses.
// The `AtomicBool` and `Semaphore` are `Send`. No shared access to `T` is created by the
// move, so `T: Sync` is not required.
unsafe impl<T: Send> Send for OnceCell<T> {}

/// Errors that can be returned from [`OnceCell::set`].
///
/// [`OnceCell::set`]: crate::sync::OnceCell::set
#[derive(Debug, PartialEq, Eq)]
pub enum SetError<T> {
    /// The cell was already initialized when [`OnceCell::set`] was called.
    ///
    /// [`OnceCell::set`]: crate::sync::OnceCell::set
    AlreadyInitializedError(T),

    /// The cell is currently being initialized.
    InitializingError(T),
}

impl<T> fmt::Display for SetError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SetError::AlreadyInitializedError(_) => write!(f, "AlreadyInitializedError"),
            SetError::InitializingError(_) => write!(f, "InitializingError"),
        }
    }
}

impl<T: fmt::Debug> Error for SetError<T> {}

impl<T> SetError<T> {
    /// Whether `SetError` is `SetError::AlreadyInitializedError`.
    pub fn is_already_init_err(&self) -> bool {
        match self {
            SetError::AlreadyInitializedError(_) => true,
            SetError::InitializingError(_) => false,
        }
    }

    /// Whether `SetError` is `SetError::InitializingError`
    pub fn is_initializing_err(&self) -> bool {
        match self {
            SetError::AlreadyInitializedError(_) => false,
            SetError::InitializingError(_) => true,
        }
    }
}
