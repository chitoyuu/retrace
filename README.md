# retrace

This crate implements a safe, generic rollback log with predictable latency.

```toml
[dependencies]
retrace = "0.1.0"
```

The `Retrace` type is implemented as a lock-free linked list of flat chunks. To keep insertion latency predictable, it simply buffers them without applying them to a "tip" state, until there is enough intents for a chunk. It then builds the chunk, prepends it to the linked list, and in most cases spawns a background task to actually produce the state of the tip. Once completed, this result is atomically put back onto the list, without affecting existing readers and speeding up subsequent queries.

This crate contains only safe code and declares `#[forbid(unsafe_code)]`.

# Use cases

This crate is written with a use case with the following characteristics in mind:

- Appends must be fast, and should not have latency spikes.
- Appends happen over time, instead of all at once.
- It is fine to waste some CPU cycles, as long as it's not blocking the main thread.

# Example

```rust
use core::fmt;
use retrace::{Retrace, Reproducible, Options};

// For the example, a simple integer is used as state. A real world example
// would have a much more complex structure in place of this.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Foo(i32);

#[derive(Clone, Debug)]
enum Never {}

impl fmt::Display for Never {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unreachable!()
    }
}

impl Reproducible for Foo {
    type Intent = i32;
    type Error = Never;

    fn apply(&mut self, intent: &Self::Intent) -> Result<(), Self::Error> {
        self.0 += *intent;
        Ok(())
    }
}

let mut retrace = Retrace::new(Foo(0), Options::default());

for i in 1..=100 {
    retrace.append(i);
}

for i in (1..=100).rev() {
    assert_eq!(
        Foo((1 + i) * i / 2),
        retrace.eval((100 - i) as usize).unwrap()
    );
}
```

# Features

- `std`: **Enabled by default.** Sets `NativeSpawner` as the default `Spawner`. Disable this default feature for `no-std` support.