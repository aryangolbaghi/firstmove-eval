"""Reproducible bounded-memory preflight benchmark for the V1 scale target."""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from queue import Empty

import psutil

from firstmove_eval.source import JsonlSource
from firstmove_eval.task import ChessFirstMoveTask

EXAMPLES = 100_000
MAX_PEAK_MIB = 128
FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _run_preflight(dataset: str, snapshot: str, results: multiprocessing.Queue[str]) -> None:
    started = time.perf_counter()
    result = JsonlSource(Path(dataset)).preflight(ChessFirstMoveTask(), Path(snapshot))
    results.put(
        json.dumps(
            {
                "valid": result.valid,
                "example_count": result.example_count,
                "elapsed": time.perf_counter() - started,
            }
        )
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="firstmove-benchmark-") as directory:
        root = Path(directory)
        dataset = root / "positions.jsonl"
        with dataset.open("w", encoding="utf-8", newline="\n") as handle:
            for index in range(EXAMPLES):
                row = {
                    "schema_version": 1,
                    "id": f"position-{index}",
                    "input": {"fen": FEN, "variant": "standard"},
                    "reference": {"move": "e4"},
                    "metadata": {},
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

        context = multiprocessing.get_context("spawn")
        results: multiprocessing.Queue[str] = context.Queue()
        process = context.Process(
            target=_run_preflight,
            args=(str(dataset), str(root / "input_snapshot.jsonl"), results),
        )
        process.start()
        monitored = psutil.Process(process.pid)
        peak = 0
        while process.is_alive():
            with suppress(psutil.Error):
                peak = max(peak, monitored.memory_info().rss)
            time.sleep(0.02)
        process.join()
        try:
            payload = json.loads(results.get(timeout=1))
        except Empty as exc:
            raise SystemExit(f"preflight worker failed with exit code {process.exitcode}") from exc

        peak_mib = peak / (1024 * 1024)
        print(
            f"examples={payload['example_count']} elapsed_s={payload['elapsed']:.2f} "
            f"peak_rss_mib={peak_mib:.2f}"
        )
        if not payload["valid"] or payload["example_count"] != EXAMPLES:
            raise SystemExit("preflight did not validate all benchmark rows")
        if peak_mib > MAX_PEAK_MIB:
            raise SystemExit(f"peak RSS {peak_mib:.2f} MiB exceeded {MAX_PEAK_MIB} MiB")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
