from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loom import explain_tailcalls, tailrec


async def hand_loop(n: int, acc: int = 0) -> int:
    while True:
        if n <= 0:
            return acc
        n, acc = n - 1, acc + 1


@tailrec
async def loom_loop(n: int, acc: int = 0) -> int:
    if n <= 0:
        return acc
    return await loom_loop(n - 1, acc + 1)


async def measure(fn, n: int, samples: int) -> tuple[float, float, float]:
    elapsed: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        result = await fn(n)
        finished = time.perf_counter()
        if result != n:
            raise AssertionError(f"{fn.__name__} returned {result}, expected {n}")
        elapsed.append(finished - started)
    return min(elapsed), statistics.mean(elapsed), max(elapsed)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Loom tail-call overhead.")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    hand_best, hand_mean, hand_worst = await measure(hand_loop, args.n, args.samples)
    loom_best, loom_mean, loom_worst = await measure(loom_loop, args.n, args.samples)
    report = explain_tailcalls(loom_loop, as_json=True)

    print(f"n={args.n}")
    print(f"samples={args.samples}")
    print(f"binding={report['binding']}")
    print(
        "hand_loop "
        f"best={hand_best:.9f}s mean={hand_mean:.9f}s worst={hand_worst:.9f}s "
        f"per_iter_best={hand_best / args.n * 1_000_000:.3f}us"
    )
    print(
        "loom_loop "
        f"best={loom_best:.9f}s mean={loom_mean:.9f}s worst={loom_worst:.9f}s "
        f"per_iter_best={loom_best / args.n * 1_000_000:.3f}us"
    )
    print(f"loom_to_hand_best_ratio={loom_best / hand_best:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
