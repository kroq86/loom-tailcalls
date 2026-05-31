import subprocess
import sys
import unittest
from pathlib import Path


class TestBenchmarkSmoke(unittest.TestCase):
    def test_benchmark_command_prints_stable_fields(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "bench_tailcalls.py"),
                "--n",
                "1000",
                "--samples",
                "1",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        output = completed.stdout
        self.assertIn("n=1000", output)
        self.assertIn("samples=1", output)
        self.assertIn("binding=direct", output)
        self.assertIn("hand_loop", output)
        self.assertIn("loom_loop", output)
        self.assertIn("loom_to_hand_best_ratio=", output)


if __name__ == "__main__":
    unittest.main()
