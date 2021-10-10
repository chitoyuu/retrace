#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(clippy::pedantic))]
#![allow(clippy::must_use_candidate, clippy::module_name_repetitions)]

extern crate alloc;

use core::fmt::{self, Debug, Display};
use core::mem::replace;
use core::ops::Deref;
use core::sync::atomic::{AtomicBool, Ordering};

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use arc_swap::{ArcSwap, ArcSwapOption, DefaultStrategy, Guard};

mod options;

/// Task spawners
pub mod spawner;

pub use options::Options;
pub use spawner::{DefaultSpawner, Spawner};

/// A generic rollback log for a [`Reproducible`] type.
///
/// The `Retrace` type is implemented as a lock-free linked list of flat chunks. To keep
/// insertion latency predictable, it simply buffers them without applying them to a "tip"
/// state, until there is enough intents for a chunk. It then builds the chunk, prepends
/// it to the linked list, and in most cases spawns a background task to actually produce
/// the state of the tip. Once completed, this result is atomically put back onto the list,
/// without affecting existing readers and speeding up subsequent queries.
///
/// The type supports one writer and multiple concurrent readers.
///
/// This type may take an extra [`Spawner`] argument that is used to spawn background tasks.
/// By default, [`DefaultSpawner`] is used.
pub struct Retrace<T: Reproducible, R = DefaultSpawner> {
    options: Options,
    spawner: Arc<R>,
    pending: Vec<T::Intent>,
    base: Option<Arc<Base<T, R>>>,
    housekeeping: AtomicBool,
}

impl<T: Reproducible> Retrace<T, DefaultSpawner> {
    /// Creates a new [`Retrace`] from a base state and a set of options, using the default
    /// spawner.
    pub fn new(state: T, options: Options) -> Self {
        Self::with_spawner(state, options, DefaultSpawner::default())
    }
}

impl<T: Reproducible, R: Spawner> Retrace<T, R> {
    /// Creates a new [`Retrace`] from a base state and a set of options, using a custom
    /// spawner.
    pub fn with_spawner(state: T, options: Options, spawner: R) -> Self {
        let pending = Vec::with_capacity(options.chunk_size.get());

        Retrace {
            options,
            spawner: Arc::new(spawner),
            pending,
            base: Some(Arc::new(Base::Root(Arc::new(state)))),
            housekeeping: AtomicBool::new(false),
        }
    }

    /// Appends a single intent to the end of the log.
    ///
    /// Appending an intent to the log does not immediately evaluate it against a "tip"
    /// state. As such, this method does not return a result. Should the appended intent
    /// be invalid (i.e. `state.apply(intent).is_err()`), queries to all subsequent states
    /// will return errors until one of the rollback methods is called to restore a known
    /// good state.
    ///
    /// This operation runs in amortized `O(1)` time, and is not dependent on the time
    /// needed to apply the intent.
    pub fn append(&mut self, intent: T::Intent) {
        self.pending.push(intent);

        let chunk_size = self.options.chunk_size.get();

        if self.pending.len() == chunk_size {
            let intents = replace(&mut self.pending, Vec::with_capacity(chunk_size));
            let intents = intents.into_boxed_slice();

            let base = self
                .base
                .take()
                .expect("base is always put back after this");

            let base_is_cached = base.is_cached();

            let chunk = Chunk::new(intents, base, Arc::clone(&self.spawner));

            if base_is_cached {
                chunk.spawn_cache_task_if_not_spawned();
            }

            self.base = Some(Arc::new(Base::Chunk(chunk)));

            self.housekeep();
        }
    }

    /// Pops a single intent off the log, if there is any.
    ///
    /// This operation runs in amortized `O(1)` time.
    pub fn pop(&mut self) -> Option<T::Intent> {
        while self.pending.is_empty() {
            if !matches!(self.base.as_deref(), Some(Base::Chunk(_))) {
                return None;
            }

            let base = self.base.take();
            let chunk = if let Some(Base::Chunk(chunk)) = base.as_deref() {
                chunk
            } else {
                unreachable!()
            };

            self.pending = chunk.intents.to_vec();
            self.base = Some(Arc::clone(&*chunk.base.load()));
        }

        self.pending.pop()
    }

    /// Returns the current length of the log.
    ///
    /// Finding out the length of the log requires walking through the (chunked) linked list,
    /// which is asymptotically `O(n)`, but usually not too large given reasonable options of
    /// `chunk_size` and `*_limit`.
    ///
    /// Note that by nature of concurrency, this value may become altered after this is called
    /// by other procedures, such as [`Retrace::housekeep`].
    pub fn len(&self) -> usize {
        let mut current = RefOrGuard::Ref(self.base.as_ref().expect("base is set here"));
        let mut len = 0;

        while let Base::Chunk(chunk) = &*current {
            len += chunk.len();
            current = RefOrGuard::Guard(chunk.base.load());
        }

        self.pending.len() + len
    }

    /// Returns `true` if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evaluates the log up to `steps` until the end, and returns the result if in range.
    ///
    /// For larger `steps` values, this requires walking through the (chunked) linked list,
    /// which is asymptotically `O(n)`, although the cost of this is likely to be dwarfed by
    /// the costs of reconstructing the state from re-applying intents, even with a cached base
    /// to start off from.
    ///
    /// # Errors
    ///
    /// - [`EvalError::OutOfRange`] if `steps` is out of range.
    /// - [`EvalError::Internal`] if an error occurred during evaluation.
    pub fn eval(&self, steps: usize) -> Result<T, EvalError<T::Error>> {
        if steps <= self.pending.len() {
            let base = self.base.as_ref().expect("base is set here");
            let mut state = base.eval().map_err(EvalError::Internal)?;
            state
                .apply_all(&self.pending[0..(self.pending.len() - steps)])
                .map_err(EvalError::Internal)?;
            return Ok(state);
        }

        let mut current = RefOrGuard::Ref(self.base.as_ref().expect("base is set here"));
        let mut steps = steps - self.pending.len();

        while let Base::Chunk(chunk) = &*current {
            if steps == 0 {
                return chunk.eval_tail().map_err(EvalError::Internal);
            } else if steps <= chunk.len() {
                return chunk.eval(steps).map_err(EvalError::Internal);
            }

            steps -= chunk.len();
            current = RefOrGuard::Guard(chunk.base.load());
        }

        Err(EvalError::OutOfRange)
    }

    /// Evaluates the "tip" state, and returns the result.
    ///
    /// This is equivalent to `retrace.eval(0)`.
    ///
    /// # Errors
    ///
    /// - [`EvalError::Internal`] if an error occurred during evaluation.
    pub fn tip(&self) -> Result<T, EvalError<T::Error>> {
        self.eval(0)
    }

    /// Attempt to find and go back to the last good state. Returns `Ok` if the current tip is
    /// already valid.
    ///
    /// Trying to find where exactly the bad intent is necessarily involves computing a lot of
    /// states, since [`Reproducible::apply`] only report the result *after* an intent is
    /// applied. For expensive intents, this is thus a lot slower than
    /// [`Retrace::rollback_chunk`] which only looks at cached results.
    ///
    /// # Errors
    ///
    /// If there is no good state to return to.
    ///
    /// # Example
    ///
    /// ```rust
    /// use retrace::{Retrace, Reproducible, Options};
    ///
    /// #[derive(Clone, PartialEq, Eq, Debug)]
    /// struct Bar(i32);
    ///
    /// impl Reproducible for Bar {
    ///     type Intent = i32;
    ///     type Error = ();
    ///
    ///     fn apply(&mut self, intent: &Self::Intent) -> Result<(), Self::Error> {
    ///         if *intent < 43 {
    ///             self.0 = *intent;
    ///             Ok(())
    ///         } else {
    ///             Err(())
    ///         }
    ///     }
    /// }
    ///
    /// let mut retrace = Retrace::new(Bar(0), Options::default());
    ///
    /// for i in 1..=100 {
    ///     retrace.append(i);
    /// }
    ///
    /// assert!(retrace.tip().is_err());
    /// assert!(retrace.rollback().is_ok());
    /// assert_eq!(Bar(42), retrace.tip().unwrap());
    /// ```
    pub fn rollback(&mut self) -> Result<(), RollbackFailure> {
        if self.tip().is_ok() {
            return Ok(());
        }

        let intents = self.rollback_chunk_inner()?;
        let mut state = self.tip().unwrap_or_else(|_| {
            unreachable!("rollback_chunk_inner should leave a valid tip or return Err")
        });

        for intent in intents {
            let mut try_state = state.clone();
            match try_state.apply(&intent) {
                Ok(()) => {
                    self.pending.push(intent);
                    state = try_state;
                }
                Err(_) => {
                    // Found the broken one
                    break;
                }
            }
        }

        debug_assert!(self.tip().is_ok());
        Ok(())
    }

    /// Attempt to find and go back to the last good state at the tail of a chunk. Returns `Ok`
    /// if the current tip is already valid.
    ///
    /// This is a lot cheaper than `rollback` since it only has to compute the tip and look at
    /// cached tail states, at the cost of losing any possibly valid intents in the next chunk.
    ///
    /// # Errors
    ///
    /// If there is no good state to return to.
    pub fn rollback_chunk(&mut self) -> Result<(), RollbackFailure> {
        if self.tip().is_ok() {
            return Ok(());
        }

        self.rollback_chunk_inner().map(|_| ())
    }

    fn rollback_chunk_inner(&mut self) -> Result<Vec<T::Intent>, RollbackFailure> {
        let mut current = RefOrGuard::Ref(self.base.as_ref().expect("base is set here"));
        let mut previous = replace(
            &mut self.pending,
            Vec::with_capacity(self.options.chunk_size.get()),
        );

        while let Base::Chunk(chunk) = &*current {
            let state = chunk.cached_tail.load();

            if let Some(state) = &*state {
                if state.is_ok() {
                    match current {
                        RefOrGuard::Ref(_) => {
                            // Still the current thing in `self.base`, nothing to do
                        }
                        RefOrGuard::Guard(base) => {
                            self.base = Some(Arc::clone(&*base));
                        }
                    }
                    return Ok(previous);
                }
            }

            previous = chunk.intents.to_vec();

            current = RefOrGuard::Guard(chunk.base.load());
        }

        if let Base::Root(state) = &*current {
            self.base = Some(Arc::new(Base::Root(Arc::new(T::clone(&*state)))));
            return Ok(Vec::new());
        }

        Err(RollbackFailure { _private: () })
    }

    /// Manually trigger housekeeping.
    ///
    /// The behavior of `housekeep()` is not specified, except that it will not cause memory
    /// unsafety, and that it will not alter any intents stored under the capacity limit,
    /// whichever is smaller among `soft_limit` and `hard_limit`.
    pub fn housekeep(&self) {
        if self
            .housekeeping
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let soft_limit = match (self.options.soft_limit, self.options.hard_limit) {
            (None, hard) => hard,
            (Some(soft), None) => Some(soft),
            (Some(soft), Some(hard)) => Some(soft.min(hard)),
        };

        let soft_limit = if let Some(soft_limit) = soft_limit {
            soft_limit
        } else {
            return;
        };

        let hard_limit = self.options.hard_limit;

        let mut current = RefOrGuard::Ref(self.base.as_ref().expect("base is set here"));
        let mut len = self.pending.len();

        while let Base::Chunk(chunk) = &*current {
            len += chunk.len();

            if hard_limit.map_or(false, |limit| len >= limit) {
                chunk.force_unchain();
                break;
            } else if len >= soft_limit {
                if chunk.unchain() {
                    break;
                }
                chunk.spawn_cache_task_if_not_spawned();
            } else if chunk.base.load().is_cached() {
                chunk.spawn_cache_task_if_not_spawned();
            }

            current = RefOrGuard::Guard(chunk.base.load());
        }

        self.housekeeping.store(false, Ordering::Release);
    }

    /// Method for benchmarking that removes cached values from all chunks. Necessary to test
    /// cold query performance.
    ///
    /// **Not** useful for any purpose other than benchmarking.
    #[cfg(feature = "bench")]
    pub fn decache(&mut self) {
        let mut current = RefOrGuard::Ref(self.base.as_ref().expect("base is set here"));
        while let Base::Chunk(chunk) = &*current {
            chunk.cached_tail.store(None);
            current = RefOrGuard::Guard(chunk.base.load());
        }
    }
}

/// Error during evaluation.
#[derive(Debug)]
#[non_exhaustive]
pub enum EvalError<E> {
    /// The requested step is not contained in the range of the log.
    OutOfRange,
    /// Any error that happened during intent application.
    Internal(E),
}

impl<E: Display> Display for EvalError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange => write!(f, "the requested step is out of range"),
            Self::Internal(e) => write!(f, "internal error: {}", e),
        }
    }
}

/// Error when there is no good state to roll back to.
#[derive(Debug)]
pub struct RollbackFailure {
    _private: (),
}

impl Display for RollbackFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no good state could be found in the log")
    }
}

/// Traits for types that are reproducible from a base value and a series of intents.
///
/// # Invariants
///
/// Correct operation of [`Retrace`] requires a number of invariants that cannot be proved by
/// the compiler. Failure to observe these invariants will not cause memory unsafety or panics,
/// but can cause erratic, undesirable behavior.
///
/// - For each value `a`, `a.clone()` is *functionally equivalent* to `a` for the use scenario.
/// - For two values `a` and `b` that are *functionally equivalent*, `a.apply(&i)` and
///   `b.apply(&i)` for the same intent `i` return the same `Result` variant, and leave the two
///   values in *functionally equivalent* state if that variant is `Ok`.
/// - The equivalency between any two values `a` and `b` is not affected by non-local state.
pub trait Reproducible: 'static + Clone + Send + Sync {
    /// The intent that can be applied to this reproducible state.
    type Intent: Clone + Send + Sync;

    /// The error type. All errors are treated as equivalent and terminate evaluation immediately.
    type Error: Clone + Send + Sync;

    /// Applies `intent` to `self` and returns the result of the operation.
    ///
    /// # Errors
    ///
    /// Implementors may return an error if `intent` is invalid, or if any other error happened
    /// during application. Returning any error from `apply` or `apply_all` short circuits
    /// evaluation  immediately, and `apply` or `apple_all` will no longer be called on this
    /// specific state again.
    fn apply(&mut self, intent: &Self::Intent) -> Result<(), Self::Error>;

    /// Apply all intents in `it` to `self` in that order, and return the result of the operation.
    ///
    /// The default implementation calls [`Reproducible::apply`] repeatedly. Some types might be
    /// able to provide efficient implementations, for example, if there is some startup or
    /// finalization cost involved.
    ///
    /// # Errors
    ///
    /// Implementors may return an error if `intent` is invalid, or if any other error happened
    /// during application. Returning any error from `apply` or `apply_all` short circuits
    /// evaluation  immediately, and `apply` or `apple_all` will no longer be called on this
    /// specific state again.
    fn apply_all<'a, I>(&mut self, it: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = &'a Self::Intent>,
    {
        for intent in it {
            self.apply(intent)?;
        }

        Ok(())
    }
}

enum RefOrGuard<'a, T: Reproducible, R> {
    Ref(&'a Base<T, R>),
    Guard(Guard<Arc<Base<T, R>>, DefaultStrategy>),
}

impl<'a, T: Reproducible, R> Deref for RefOrGuard<'a, T, R> {
    type Target = Base<T, R>;
    fn deref(&self) -> &Self::Target {
        match self {
            RefOrGuard::Ref(r) => r,
            RefOrGuard::Guard(r) => &**r,
        }
    }
}

struct Chunk<T: Reproducible, R> {
    spawner: Arc<R>,
    cached_tail: Arc<ArcSwapOption<Result<Arc<T>, T::Error>>>,
    cache_task_spawned: AtomicBool,
    intents: Arc<[T::Intent]>,
    base: ArcSwap<Base<T, R>>,
}

impl<T: Reproducible, R: Spawner> Chunk<T, R> {
    fn new(intents: Box<[T::Intent]>, base: Arc<Base<T, R>>, spawner: Arc<R>) -> Self {
        Chunk {
            spawner,
            cached_tail: Arc::new(ArcSwapOption::from_pointee(None)),
            cache_task_spawned: AtomicBool::new(false),
            intents: Arc::from(intents),
            base: ArcSwap::new(base),
        }
    }

    fn spawn_cache_task_if_not_spawned(&self) {
        if self
            .cache_task_spawned
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        let base = self.base.load();
        let cached_tail = Arc::clone(&self.cached_tail);
        let intents = Arc::clone(&self.intents);

        self.spawner.spawn(move || {
            let result = base.eval().and_then(move |mut state| {
                state.apply_all(&*intents)?;
                Ok(state)
            });

            cached_tail.store(Some(Arc::new(result.map(Arc::new))));
        });
    }

    /// Unchain from this chunk if there is a cached base value available, making this chunk the new
    /// root. Returns whether the current chunk is now a root chunk.
    fn unchain(&self) -> bool {
        let base = self.base.load();

        let chunk = match &**base {
            Base::Chunk(chunk) => chunk,
            Base::Root(_) | Base::Error(_) => return true,
        };

        let result = if let Some(result) = chunk.cached_tail.swap(None) {
            result
        } else {
            return false;
        };

        let base = match &*result {
            Ok(t) => Base::Root(Arc::clone(t)),
            Err(e) => Base::Error(T::Error::clone(e)),
        };

        let old = self.base.swap(Arc::new(base));

        // Send the drop to another thread for the operation is potentially expensive.
        self.spawner.spawn(move || drop(old));

        true
    }

    /// Unchain from this chunk, evaluating on the current thread if necessary.
    fn force_unchain(&self) {
        let base = self.base.load();

        let chunk = match &**base {
            Base::Chunk(chunk) => chunk,
            Base::Root(_) | Base::Error(_) => return,
        };

        let base = match chunk.eval_tail() {
            Ok(t) => Base::Root(Arc::new(t)),
            Err(e) => Base::Error(e),
        };

        let old = self.base.swap(Arc::new(base));

        // Send the drop to another thread for the operation is potentially expensive.
        self.spawner.spawn(move || drop(old));
    }

    fn len(&self) -> usize {
        self.intents.len()
    }

    fn eval_tail(&self) -> Result<T, T::Error> {
        let cached = self.cached_tail.load();

        if let Some(result) = &*cached {
            return match &**result {
                Ok(t) => Ok(T::clone(t)),
                Err(e) => Err(T::Error::clone(e)),
            };
        }

        let result = self.eval(0);

        self.cached_tail.compare_and_swap(
            cached,
            Some(Arc::new(match &result {
                Ok(t) => Ok(Arc::new(T::clone(t))),
                Err(e) => Err(T::Error::clone(e)),
            })),
        );

        result
    }

    /// Evaluate the current chunk up to `steps` until the end.
    ///
    /// # Panics
    ///
    /// If `steps` is larger than `intents.len()`
    fn eval(&self, steps: usize) -> Result<T, T::Error> {
        let base = self.base.load();
        let mut state = base.eval()?;

        let intents = self.intents.as_ref();
        assert!(steps <= intents.len());

        state.apply_all(&intents[0..(intents.len() - steps)])?;

        Ok(state)
    }
}

enum Base<T: Reproducible, R> {
    Root(Arc<T>),
    Chunk(Chunk<T, R>),
    Error(T::Error),
}

impl<T: Reproducible, R: Spawner> Base<T, R> {
    fn is_cached(&self) -> bool {
        match self {
            Base::Root(_) | Base::Error(_) => true,
            Base::Chunk(chunk) => chunk.cached_tail.load().is_some(),
        }
    }

    fn eval(&self) -> Result<T, T::Error> {
        match self {
            Base::Root(t) => Ok(T::clone(t)),
            Base::Error(e) => Err(T::Error::clone(e)),
            Base::Chunk(chunk) => chunk.eval_tail(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::num::NonZeroUsize;

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct Foo(i32);

    impl Reproducible for Foo {
        type Intent = i32;
        type Error = ();

        fn apply(&mut self, intent: &Self::Intent) -> Result<(), Self::Error> {
            self.0 += *intent;
            Ok(())
        }
    }

    #[test]
    fn is_send_sync() {
        fn test<T: Send + Sync>() {}
        test::<Foo>();
        test::<Retrace<Foo, DefaultSpawner>>();
        test::<Chunk<Foo, DefaultSpawner>>();
        test::<Base<Foo, DefaultSpawner>>();
    }

    #[test]
    fn basic_functions() {
        const MAX: i32 = 100;

        let mut retrace = Retrace::new(
            Foo(0),
            Options::new().chunk_size(NonZeroUsize::new(4).unwrap()),
        );

        for i in 1..=MAX {
            retrace.append(i);
        }

        for _ in 0..MAX {
            retrace.append(42);
        }

        for _ in 0..MAX {
            assert_eq!(Some(42), retrace.pop());
        }

        for i in (1..=MAX).rev() {
            assert_eq!(
                Foo((1 + i) * i / 2),
                retrace.eval((MAX - i) as usize).unwrap()
            );
        }

        assert!(matches!(
            retrace.eval((MAX + 1) as usize).unwrap_err(),
            EvalError::OutOfRange
        ));

        for i in (1..=MAX).rev() {
            assert_eq!(Some(i), retrace.pop());
        }

        assert_eq!(None, retrace.pop());
    }

    #[test]
    fn backward_query() {
        const MAX: i32 = 100;

        let mut retrace = Retrace::new(
            Foo(0),
            Options::new().chunk_size(NonZeroUsize::new(4).unwrap()),
        );

        for i in 1..=MAX {
            retrace.append(i);
        }

        for i in 1..=MAX {
            assert_eq!(
                Foo((1 + i) * i / 2),
                retrace.eval((MAX - i) as usize).unwrap()
            );
        }
    }

    #[test]
    fn no_force_rebase_for_soft_limit() {
        let mut retrace = Retrace::with_spawner(
            Foo(0),
            Options::new()
                .chunk_size(NonZeroUsize::new(4).unwrap())
                .soft_limit(Some(20)),
            spawner::DropSpawner::default(),
        );

        for _ in 0..100 {
            retrace.append(42);
        }

        assert_eq!(100, retrace.len());
    }

    #[test]
    fn force_rebase_for_hard_limit() {
        let mut retrace = Retrace::with_spawner(
            Foo(0),
            Options::new()
                .chunk_size(NonZeroUsize::new(4).unwrap())
                .hard_limit(Some(20)),
            spawner::DropSpawner::default(),
        );

        for _ in 0..20 {
            retrace.append(42);
        }

        for _ in 0..100 {
            retrace.append(42);
            assert!((20..=24).contains(&retrace.len()));
        }
    }

    #[test]
    fn deferred_rebase_for_soft_limit() {
        let mut retrace = Retrace::with_spawner(
            Foo(0),
            Options::new()
                .chunk_size(NonZeroUsize::new(4).unwrap())
                .soft_limit(Some(20)),
            spawner::ImmediateSpawner::default(),
        );

        for _ in 0..100 {
            retrace.append(42);
        }

        assert!((20..=24).contains(&retrace.len()));
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct Bar(i32);

    impl Reproducible for Bar {
        type Intent = Option<i32>;
        type Error = ();

        fn apply(&mut self, intent: &Self::Intent) -> Result<(), Self::Error> {
            match intent {
                Some(intent) => {
                    self.0 += *intent;
                    Ok(())
                }
                None => Err(()),
            }
        }
    }

    #[test]
    fn can_rollback_chunk() {
        let mut retrace = Retrace::new(
            Bar(0),
            Options::new().chunk_size(NonZeroUsize::new(4).unwrap()),
        );

        for _ in 0..22 {
            retrace.append(Some(1));
        }

        retrace.append(None);

        for _ in 0..22 {
            retrace.append(Some(1));
        }

        assert!(retrace.tip().is_err());
        retrace.rollback_chunk().unwrap();
        assert_eq!(Bar(20), retrace.tip().unwrap());
    }

    #[test]
    fn can_rollback_precise() {
        let mut retrace = Retrace::new(
            Bar(0),
            Options::new().chunk_size(NonZeroUsize::new(4).unwrap()),
        );

        for _ in 0..22 {
            retrace.append(Some(1));
        }

        retrace.append(None);

        for _ in 0..22 {
            retrace.append(Some(1));
        }

        assert!(retrace.tip().is_err());
        retrace.rollback().unwrap();
        assert_eq!(Bar(22), retrace.tip().unwrap());
    }

    #[test]
    fn can_rollback_to_root() {
        let mut retrace = Retrace::new(
            Bar(0),
            Options::new().chunk_size(NonZeroUsize::new(4).unwrap()),
        );

        retrace.append(None);

        assert!(retrace.tip().is_err());
        retrace.rollback().unwrap();
        assert_eq!(Bar(0), retrace.tip().unwrap());

        let mut retrace = Retrace::new(
            Bar(0),
            Options::new().chunk_size(NonZeroUsize::new(4).unwrap()),
        );

        retrace.append(None);

        assert!(retrace.tip().is_err());
        retrace.rollback_chunk().unwrap();
        assert_eq!(Bar(0), retrace.tip().unwrap());
    }
}
