use std::num::NonZeroUsize;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

use retrace::spawner::DropSpawner;
use retrace::{Options, Reproducible, Retrace};

#[derive(Clone, Default)]
struct Wrap<T>(T);

impl<const N: usize> Reproducible for Wrap<[u8; N]> {
    type Intent = [u8; N];
    type Error = ();
    fn apply(&mut self, intent: &Self::Intent) -> Result<(), Self::Error> {
        if intent.iter().all(|&byte| byte == 0) {
            return Err(());
        }

        // Some artificial sisyphean task
        for split in (0..N).step_by(4) {
            for (a, b) in self
                .0
                .iter_mut()
                .zip(intent[split..N].iter().chain(&intent[0..split]))
            {
                *a ^= b.rotate_right((split & 7) as u32);
            }
        }

        Ok(())
    }
}

fn make_intents<const N: usize>(amount: usize, valid: bool) -> Vec<[u8; N]> {
    let mut rng = rand::thread_rng();
    let mut v = Vec::with_capacity(amount);
    for _ in 0..amount {
        let mut intent = [0u8; N];
        rng.fill(intent.as_mut());
        if valid {
            while intent == [0u8; N] {
                rng.fill(intent.as_mut());
            }
        }

        v.push(intent);
    }
    v
}

fn bench_insertion<const N: usize>(c: &mut Criterion, amount: usize, chunk_size: usize) {
    let name = format!("random insert {}x{}B chunk_size {}", amount, N, chunk_size);

    c.bench_function(&name, |b| {
        let intents = make_intents(amount, false);
        let chunk_size = NonZeroUsize::new(chunk_size).unwrap();

        let mut init_state = [0u8; N];
        rand::thread_rng().fill(init_state.as_mut());

        b.iter(|| {
            let mut retrace = Retrace::with_spawner(
                Wrap(black_box(init_state)),
                Options::default().chunk_size(chunk_size),
                DropSpawner::default(),
            );

            for intent in black_box(&intents) {
                retrace.append(*intent);
            }
        })
    });
}

fn bench_insertion_chunk_sizes<const N: usize>(c: &mut Criterion) {
    bench_insertion::<N>(c, 1000, 32);
    bench_insertion::<N>(c, 1000, 128);
}

fn bench_query<const N: usize>(c: &mut Criterion, amount: usize, chunk_size: usize, cold: bool) {
    let name = if cold {
        format!("random query {}x{}B chunk_size {}", amount, N, chunk_size)
    } else {
        format!(
            "random query warm {}x{}B chunk_size {}",
            amount, N, chunk_size
        )
    };

    c.bench_function(&name, |b| {
        let mut rng = rand::thread_rng();
        let chunk_size = NonZeroUsize::new(chunk_size).unwrap();

        let mut init_state = [0u8; N];
        rng.fill(init_state.as_mut());

        let mut retrace = Retrace::with_spawner(
            Wrap(black_box(init_state)),
            Options::default().chunk_size(chunk_size),
            DropSpawner::default(),
        );

        for intent in make_intents(amount, true) {
            retrace.append(intent);
        }

        b.iter(|| {
            if cold {
                retrace.decache();
            }
            black_box(retrace.eval(rng.gen_range(black_box(0..amount)))).ok();
        })
    });
}

fn bench_query_chunk_sizes<const N: usize>(c: &mut Criterion) {
    bench_query::<N>(c, 1000, 32, true);
    bench_query::<N>(c, 1000, 128, true);
    bench_query::<N>(c, 1000, 32, false);
    bench_query::<N>(c, 1000, 128, false);
}

fn bench_raw_apply<const N: usize>(c: &mut Criterion) {
    let name = format!("apply on {}B intents", N);

    c.bench_function(&name, |b| {
        let mut rng = rand::thread_rng();

        let mut state = [0u8; N];
        rng.fill(state.as_mut());

        let mut intent = [0u8; N];
        rng.fill(intent.as_mut());

        b.iter(|| Wrap(black_box(state)).apply(black_box(&intent)))
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_insertion_chunk_sizes::<64>(c);
    bench_insertion_chunk_sizes::<256>(c);
    bench_insertion_chunk_sizes::<1024>(c);

    bench_raw_apply::<64>(c);
    bench_raw_apply::<256>(c);
    bench_raw_apply::<1024>(c);

    bench_query_chunk_sizes::<64>(c);
    bench_query_chunk_sizes::<256>(c);
    bench_query_chunk_sizes::<1024>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
