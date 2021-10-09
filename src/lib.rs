#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use core::marker::PhantomData;
use core::slice::SliceIndex;
use core::num::NonZeroUsize;

use alloc::sync::Arc;
use alloc::vec::Vec;

use arc_swap::{ArcSwap, ArcSwapOption};

/// A generic rollback log for any [`Reproducible`] type.
pub struct Retrace<T: Reproducible, R = DefaultSpawner> {
    options: Options,
    spawner: Arc<R>,
    pending: Vec<T::Intent>,
    base: Base<T, R>,
}

impl<T: Reproducible, R: Spawner + Default> Retrace<T, R> {
    pub fn new(state: T, options: &Options) -> Self {
        Self::with_spawner(state, options, R::default())
    }
}

impl<T: Reproducible, R: Spawner> Retrace<T, R> {
    pub fn with_spawner(state: T, options: &Options, spawner: R) -> Self {
        Retrace {
            options: options.clone(),
            spawner: Arc::new(spawner),
            pending: Vec::with_capacity(options.chunk_size.get()),
            base: Base::Root(Arc::new(state)),
        }
    }
}

/// Options for creating a new instance of [`Retrace`].
///
/// This type supports builder-style methods. The default can be obtained using either
/// [`Options::default`] or [`Options::new`]. The default values are subject to change between
/// compatible versions.
#[derive(Clone, Debug)]
pub struct Options {
    chunk_size: NonZeroUsize,
    soft_limit: Option<usize>,
    hard_limit: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            chunk_size: NonZeroUsize::new(32).unwrap(),
            soft_limit: Some(256),
            hard_limit: None,
        }
    }
}

impl Options {
    /// Creates a new [`Options`] from defaults. Same as [`Options::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the chunk size for this instance. At most one extra copy of the state can be cached
    /// every `chunk_size` logged intents. This value must be non-zero.
    ///
    /// The value of `chunk_size` represents a tradeoff between retrieval latency and memory
    /// efficiency. On one hand, for large states and relatively inexpensive intents, you might
    /// want a large `chunk_size` to avoid storing many copies of the state. On the other hand,
    /// if the states are small and the intents are expensive, a smaller `chunk_size` might be
    /// useful.
    ///
    /// The current default value is `32`.
    pub fn chunk_size(&mut self, chunk_size: NonZeroUsize) -> &mut Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Sets a soft limit on history capacity for this instance.
    ///
    /// [`Retrace`] instances can potentially grow to store any number of intents as memory
    /// allows. In practice, you might only ever need to revert a small amount of intents from
    /// the tip. A `soft_limit` value hints [`Retrace`] to consider dropping trailing chunks
    /// opportunistically when the length of the log becomes longer than the number specified.
    ///
    /// [`Retrace`] will *never* block the calling thread to rebase to `soft_limit`. A
    /// `soft_limit` may never do anything if background tasks are dropped with [`DropSpawner`].
    /// See [`Options::hard_limit`] for a complement.
    ///
    /// The current default value is `None`.
    pub fn soft_limit(&mut self, soft_limit: Option<usize>) -> &mut Self {
        self.soft_limit = soft_limit;
        self
    }
    /// Sets a hard limit on history capacity for this instance.
    ///
    /// Unlike `soft_limit`, [`Retrace`] *will* block the calling thread in order to rebase
    /// whenever it accumulates extra capacity over `hard_limit`. This can cause regular spikes
    /// in latency in [`Retrace::push`], depending on how expensive it is to evaluate new base
    /// states.
    ///
    /// `hard_limit` is most useful when combined with a lower `soft_limit` that allows
    /// [`Retrace`] to opportunistically truncate old states, which has a much more predictable
    /// performance.
    ///
    /// The current default value is `None`.
    pub fn hard_limit(&mut self, hard_limit: Option<usize>) -> &mut Self {
        self.hard_limit = hard_limit;
        self
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
/// - For each value `a`, `a.clone()` is *functionally equivalent* for the use scenario.
/// - For two values `a` and `b` that are *functionally equivalent*, `a.apply(&i)` and
///   `b.apply(&i)` for the same intent `i` return the same `Result` variant, and leave the two
///   values in *functionally equivalent* state if that variant is `Ok`.
pub trait Reproducible: 'static + Clone + Send + Sync {
    /// The intent that can be applied to this reproducible state.
    type Intent: Send + Sync;

    /// The error type.
    type Error: Clone + Send + Sync;

    /// Applies `intent` to `self` and returns the result of the operation.
    fn apply(&mut self, intent: &Self::Intent) -> Result<(), Self::Error>;
}

/// The default task spawner used by [`Retrace`] if the parameter is omitted.
///
/// This is currently [`NativeSpawner`] if `std` is enabled, or [`DropSpawner`] if not.
#[cfg(feature = "std")]
pub type DefaultSpawner = NativeSpawner;

/// The default task spawner used by [`Retrace`] if the parameter is omitted.
///
/// This is currently [`NativeSpawner`] if `std` is enabled, or [`DropSpawner`] if not.
#[cfg(not(feature = "std"))]
pub type DefaultSpawner = DropSpawner;

/// Trait for any spawner that supports spawning an unsupervised background task.
///
/// [`Retrace`] only uses the spawner for non-essential tasks. It does not depend on the
/// closures actually being run to function as expected. See [`DropSpawner`] for a spawner
/// that simply drops everything passed to it.
///
/// The default spawner is available as the type alias [`DefaultSpawner`].
pub trait Spawner: 'static + Send + Sync {
    /// Spawns the task `f` on another thread
    fn spawn<F>(&self, f: F)
    where
        F: 'static + FnOnce() + Send;
}

#[cfg(feature = "std")]
pub use native_spawner::NativeSpawner;

#[cfg(feature = "std")]
mod native_spawner {
    use super::Spawner;

    /// A task spawner that delegates to [`std::thread::spawn`].
    #[derive(Clone, Copy, Debug, Default)]
    pub struct NativeSpawner {
        _private: (),
    }

    impl Spawner for NativeSpawner {
        fn spawn<F>(&self, f: F)
        where
            F: 'static + FnOnce() + Send,
        {
            std::thread::spawn(f);
        }
    }
}

/// A task "spawner" that simply drops the closure immediately on the current thread.
///
/// This can be used to disable all background tasks, when they are undesirable or if an actual
/// spawner is unavailable.
#[derive(Clone, Copy, Debug, Default)]
pub struct DropSpawner {
    _private: (),
}

impl Spawner for DropSpawner {
    fn spawn<F>(&self, _f: F)
    where
        F: 'static + FnOnce() + Send,
    {
    }
}

struct Chunk<T: Reproducible, R> {
    spawner: Arc<R>,
    cached_tail: Arc<ArcSwapOption<Result<Arc<T>, T::Error>>>,
    intents: Box<[T::Intent]>,
    base: ArcSwap<Base<T, R>>,
}

impl<T: Reproducible, R: Spawner> Chunk<T, R> {
    fn new(intents: Box<[T::Intent]>, base: Base<T, R>, spawner: Arc<R>) -> Self {
        Chunk {
            spawner,
            cached_tail: Arc::new(ArcSwapOption::from_pointee(None)),
            intents,
            base: ArcSwap::from_pointee(base),
        }
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

        let result = self.eval(..);

        self.cached_tail.compare_and_swap(
            cached,
            Some(Arc::new(match &result {
                Ok(t) => Ok(Arc::new(T::clone(t))),
                Err(e) => Err(T::Error::clone(e)),
            })),
        );

        result
    }

    fn eval<I>(&self, range: I) -> Result<T, T::Error>
    where
        I: SliceIndex<[T::Intent], Output = [T::Intent]>,
    {
        let base = self.base.load();
        let mut state = base.eval()?;
        for intent in &self.intents.as_ref()[range] {
            state.apply(intent)?;
        }
        Ok(state)
    }
}

enum Base<T: Reproducible, R> {
    Root(Arc<T>),
    Chunk(Chunk<T, R>),
    Error(T::Error),
}

impl<T: Reproducible, R: Spawner> Base<T, R> {
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

    #[derive(Clone)]
    struct Foo(i32);

    #[derive(Clone)]
    enum Never {}

    impl Reproducible for Foo {
        type Intent = i32;
        type Error = Never;

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
}
