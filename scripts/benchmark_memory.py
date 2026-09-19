"""Reproducible CPU comparison of individual and batched gallery searches."""
import argparse
import contextlib
import io
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Examples:\n  python -m scripts.benchmark_memory --json\n"
               "  python -m scripts.benchmark_memory --labels 1000 --queries 100 --json > results.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--labels", type=positive, default=100)
    parser.add_argument("--examples", type=positive, default=16, help="examples per label")
    parser.add_argument("--dimension", type=positive, default=768)
    parser.add_argument("--queries", type=positive, default=10)
    parser.add_argument("--samples", type=positive, default=20)
    parser.add_argument("--warmups", type=positive, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", type=Path, default=Path("data"), help="parent of an isolated temporary database")
    parser.add_argument("--json", action="store_true", help="emit measurements and environment metadata as JSON")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("use a non-negative --seed")
    if args.labels > 1000 or args.examples > 16:
        parser.error("use --labels <= 1000 and --examples <= 16 (application gallery limits)")
    if args.dimension > 4096 or args.queries > 1000:
        parser.error("use --dimension <= 4096 and --queries <= 1000")

    import numpy as np

    from odin_vision.config import Settings
    from odin_vision.memory import Memory

    rng = np.random.default_rng(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="odin-benchmark-", dir=args.work_dir.resolve()) as directory:
        if Path(directory).resolve().parent != args.work_dir.resolve():
            raise RuntimeError("temporary database is outside --work-dir")
        memory = Memory(Settings(data_dir=Path(directory)), "synthetic-benchmark-v1")
        try:
            references = []
            for index in range(args.labels):
                vectors = rng.normal(size=(args.examples, args.dimension)).astype(np.float32)
                memory.learn(f"label-{index}", "object", vectors)
                references.append(vectors[0])
            queries = rng.normal(size=(args.queries, args.dimension)).astype(np.float32)
            # Mix known and unknown probes; neither timings nor agreement imply identity accuracy.
            for index in range(0, args.queries, 2):
                queries[index] = references[index % len(references)]
            functions = {
                "individual": lambda: [memory.match(vector, "object", .91, .02) for vector in queries],
                "batch": lambda: memory.match_many(queries, "object", .91, .02),
            }
            expected = functions["individual"]()
            samples = {name: [] for name in functions}
            for iteration in range(args.warmups + args.samples):
                order = list(functions) if iteration % 2 else list(reversed(functions))
                for name in order:
                    start = time.perf_counter()
                    result = functions[name]()
                    elapsed = (time.perf_counter()-start)*1000
                    if result != expected:
                        raise RuntimeError("individual/batch result mismatch, including exact scores")
                    if iteration >= args.warmups:
                        samples[name].append(elapsed)
            examples = memory.db.execute("SELECT count(*) FROM examples").fetchone()[0]
        finally:
            memory.close()
    config_output = io.StringIO()
    with contextlib.redirect_stdout(config_output):
        np.show_config()
    medians = {name: statistics.median(values) for name, values in samples.items()}
    report = {
        "schema_version": 1,
        "scope": "synthetic cached-gallery CPU search; database creation and loading excluded",
        "parameters": {key: value for key, value in vars(args).items() if key not in {"json", "work_dir"}},
        "stored_examples": examples,
        "accepted_queries": sum(label is not None for label, _ in expected),
        "results_and_scores_exact": True,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "processor": platform.processor(),
                        "numpy_config": config_output.getvalue()},
        "samples_ms": samples,
        "median_ms": medians,
        "median_speedup": medians["individual"] / medians["batch"],
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Individual {medians['individual']:.3f} ms; batch {medians['batch']:.3f} ms; "
              f"{report['median_speedup']:.2f}x; exact results for {args.queries} queries.")


if __name__ == "__main__":
    main()
