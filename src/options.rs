use core::num::NonZeroUsize;

/// Options for creating a new instance of Retrace.
///
/// This type supports builder-style methods. The default can be obtained using either
/// [`Options::default`] or [`Options::new`]. The default values are subject to change between
/// compatible versions.
#[derive(Clone, Debug)]
pub struct Options {
    pub(crate) chunk_size: NonZeroUsize,
    pub(crate) soft_limit: Option<usize>,
    pub(crate) hard_limit: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            chunk_size: NonZeroUsize::new(32).unwrap(),
            soft_limit: None,
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
    /// For allocation purposes, for any intent type `I`, this is best set so that
    /// `chunk_size * size_of::<I>()` either is, or is close to the lower side of a power-of-2.
    ///
    /// The current default value is `32`.
    pub fn chunk_size(mut self, chunk_size: NonZeroUsize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Sets a soft limit on history capacity for this instance.
    ///
    /// Retrace instances can potentially grow to store any number of intents as memory
    /// allows. In practice, you might only ever need to revert a small amount of intents from
    /// the tip. A `soft_limit` value hints Retrace to consider dropping trailing chunks
    /// opportunistically when the length of the log becomes longer than the number specified.
    ///
    /// Retrace will *never* block the calling thread to rebase to `soft_limit`. A
    /// `soft_limit` may never do anything if background tasks are dropped with
    /// [`crate::spawner::DropSpawner`]. See [`Options::hard_limit`] for a complement.
    ///
    /// The current default value is `None`.
    pub fn soft_limit(mut self, soft_limit: Option<usize>) -> Self {
        self.soft_limit = soft_limit;
        self
    }
    /// Sets a hard limit on history capacity for this instance.
    ///
    /// Unlike `soft_limit`, Retrace *will* block the calling thread in order to rebase
    /// whenever it accumulates extra capacity over `hard_limit`. This can cause regular spikes
    /// in latency, depending on how expensive it is to evaluate new base states.
    ///
    /// `hard_limit` is most useful when combined with a lower `soft_limit` that allows
    /// Retrace to opportunistically truncate old states, which has a much more predictable
    /// performance.
    ///
    /// The current default value is `None`.
    pub fn hard_limit(mut self, hard_limit: Option<usize>) -> Self {
        self.hard_limit = hard_limit;
        self
    }
}
