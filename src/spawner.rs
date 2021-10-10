/// The default task spawner used by Retrace if the parameter is omitted.
///
/// This is currently [`NativeSpawner`] if `feature = "std"` is enabled, or
/// [`DropSpawner`] if not.
#[cfg(feature = "std")]
pub type DefaultSpawner = NativeSpawner;

/// The default task spawner used by Retrace if the parameter is omitted.
///
/// This is currently [`NativeSpawner`] if `feature = "std"` is enabled, or
/// [`DropSpawner`] if not.
#[cfg(not(feature = "std"))]
pub type DefaultSpawner = DropSpawner;

/// Trait for any spawner that supports spawning an unsupervised background task.
///
/// Retrace only uses the spawner for non-essential tasks. It does not depend on the
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

/// A task "spawner" that simply drops the closure.
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

/// A task spawner that invokes the closure immediately on the current thread. This is usually a
/// *very bad idea* except in tests, where its deterministic behavior can be helpful.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImmediateSpawner {
    _private: (),
}

impl Spawner for ImmediateSpawner {
    fn spawn<F>(&self, f: F)
    where
        F: 'static + FnOnce() + Send,
    {
        f();
    }
}
