"""The `goldenset` command line."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

EVAL_CHOICES = ("scrub", "dedup", "coverage", "enrich", "review")


# --- generate / make-seed ---


def _add_generate(sub: Any) -> None:
    parser = sub.add_parser("generate", help="Generate a synthetic trace dataset with ground truth")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(func=_cmd_generate)


def _cmd_generate(args: argparse.Namespace) -> int:
    from goldenset.generator.build import generate_dataset, write_dataset

    write_dataset(generate_dataset(args.n, args.seed), args.out)
    print(f"Wrote {args.n} traces to {args.out}")
    return 0


def _add_make_seed(sub: Any) -> None:
    parser = sub.add_parser("make-seed", help="Build the seed golden set from a generated dataset")
    parser.add_argument("--data", type=Path, required=True, help="Directory with traces.jsonl and ground_truth.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(func=_cmd_make_seed)


def _cmd_make_seed(args: argparse.Namespace) -> int:
    from goldenset.generator.truth import load_truths
    from goldenset.intake import load_valid_traces
    from goldenset.jsonl import write_jsonl
    from goldenset.seed import build_seed_golden, pick_seed_ids
    from goldenset.stages.scrub_engine import ScrubEngine

    traces = load_valid_traces(args.data / "traces.jsonl")
    truths = load_truths(args.data / "ground_truth.jsonl")
    ids = pick_seed_ids(truths.values(), args.count, args.seed)
    records, skipped = build_seed_golden(traces, ids, truths, ScrubEngine())
    write_jsonl(args.out, records)
    print(f"Wrote {len(records)} seed records to {args.out} ({len(skipped)} skipped for remaining PII)")
    return 0


# --- run ---


def _add_run(sub: Any) -> None:
    parser = sub.add_parser("run", help="Run the pipeline (resumable)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stage", help="Run only this stage")
    group.add_argument("--until", help="Run every stage up to and including this one")
    parser.add_argument("--force", action="store_true", help="Re-run stages even if their output is current")
    parser.set_defaults(func=_cmd_run)


def _cmd_run(args: argparse.Namespace) -> int:
    from goldenset.config import load_config
    from goldenset.embeddings import EmbedderUnavailable
    from goldenset.evaluation.report import funnel_table, write_report
    from goldenset.pipeline.build import build_services, build_stages
    from goldenset.pipeline.runner import Runner, StageFailed

    config = load_config(args.config)
    run_dir = args.runs_dir / (args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        runner = Runner(config, run_dir, build_stages(), build_services(config))
        stats = runner.run(only=args.stage, force=args.force, until=args.until)
    except (StageFailed, EmbedderUnavailable) as exc:
        print(exc, file=sys.stderr)
        runs_dir = f" --runs-dir {args.runs_dir}" if args.runs_dir != Path("runs") else ""
        print(f"Resume with: goldenset run --config {args.config} --run-id {run_dir.name}{runs_dir}", file=sys.stderr)
        return 1
    write_report(run_dir)
    print(funnel_table(stats))
    print(f"\nRun folder: {run_dir}")
    return 0


# --- eval / calibrate / report ---


def _add_eval(sub: Any) -> None:
    parser = sub.add_parser("eval", help="Measure stages against ground truth")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--what", default=",".join(EVAL_CHOICES), help="Comma-separated subset of " + ", ".join(EVAL_CHOICES))
    parser.add_argument("--limit", type=int, help="Scrub evaluation only: use the first N traces")
    parser.add_argument("--scrub-threshold", type=float, help="Scrub evaluation only: override the detection threshold")
    parser.set_defaults(func=_cmd_eval)


def _write_eval(run_dir: Path, name: str, payload: dict[str, Any]) -> None:
    out = run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out / (name + '.json')}")


def _cmd_eval(args: argparse.Namespace) -> int:
    from itertools import islice

    from goldenset.evaluation.coverage_eval import evaluate_coverage, stage_ids
    from goldenset.evaluation.dedup_eval import evaluate_dedup
    from goldenset.evaluation.enrich_eval import evaluate_enrichment
    from goldenset.evaluation.report import write_report
    from goldenset.evaluation.review_eval import evaluate_review
    from goldenset.evaluation.scrub_eval import evaluate_scrub
    from goldenset.generator.truth import load_truths
    from goldenset.intake import load_valid_traces
    from goldenset.jsonl import read_jsonl
    from goldenset.models import Record
    from goldenset.stages.scrub_engine import ScrubEngine

    run_dir: Path = args.run
    saved = json.loads((run_dir / "config.json").read_text())["config"]
    truths = load_truths(args.truth)
    wanted = [w.strip() for w in args.what.split(",") if w.strip()]
    unknown = set(wanted) - set(EVAL_CHOICES)
    if unknown:
        print(f"Unknown evaluation(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    for what in wanted:
        if what == "scrub":
            threshold = args.scrub_threshold if args.scrub_threshold is not None else saved["scrub"]["threshold"]
            traces = load_valid_traces(Path(saved["input_path"])).values()
            subset = list(islice(traces, args.limit) if args.limit else traces)
            engine = ScrubEngine(saved["scrub"]["spacy_model"], threshold)
            name = "scrub" if args.scrub_threshold is None else f"scrub-t{args.scrub_threshold}"
            _write_eval(run_dir, name, evaluate_scrub(subset, truths, engine).summary())
        elif what == "dedup":
            decisions = list(read_jsonl(run_dir / "decisions.jsonl"))
            input_ids = stage_ids(run_dir / "stages" / "03_scrub.jsonl")
            _write_eval(run_dir, "dedup", evaluate_dedup(decisions, truths, input_ids))
        elif what == "coverage":
            _write_eval(run_dir, "coverage", evaluate_coverage(run_dir, truths))
        elif what == "enrich":
            records = [Record.model_validate(r) for r in read_jsonl(run_dir / "stages" / "06_enrich.jsonl")]
            result = evaluate_enrichment(records, truths)
            result["baseline"] = _keyword_baseline(run_dir, saved, records, truths)
            _write_eval(run_dir, "enrich", result)
        elif what == "review":
            _write_eval(run_dir, "review", evaluate_review(run_dir, saved["review"]["hourly_rate_usd"]))
    write_report(run_dir)
    return 0


def _keyword_baseline(run_dir: Path, saved: dict[str, Any], records: list[Any], truths: Any) -> dict[str, Any]:
    """Label the same records with the free keyword baseline, to put the LLM's accuracy in context."""
    import tempfile

    from goldenset.config import RunConfig
    from goldenset.evaluation.enrich_eval import evaluate_enrichment
    from goldenset.llm import RuleBasedLLM
    from goldenset.pipeline.context import DecisionLog, Services, StageContext
    from goldenset.stages.enrich import EnrichStage

    with tempfile.TemporaryDirectory() as scratch:  # its own log: the run's decisions.jsonl stays untouched
        ctx = StageContext(
            "enrich-baseline", run_dir, RunConfig.model_validate(saved), DecisionLog(Path(scratch) / "d.jsonl"),
            Services(llm=RuleBasedLLM()),
        )
        fresh = [r.model_copy(deep=True) for r in records]
        for record in fresh:
            record.meta.enrichment = None
        labelled = list(EnrichStage().run(iter(fresh), ctx))
    return evaluate_enrichment(labelled, truths)


def _add_calibrate(sub: Any) -> None:
    parser = sub.add_parser("calibrate", help="Choose dedup and novelty thresholds from ground truth")
    parser.add_argument("--run", type=Path, required=True, help="A run that has completed the scrub stage")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--write", type=Path, help="Config file to update with the chosen thresholds")
    parser.set_defaults(func=_cmd_calibrate)


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from goldenset.config import RunConfig
    from goldenset.embeddings import make_embedder
    from goldenset.evaluation.calibrate import calibrate_run
    from goldenset.generator.truth import load_truths

    saved = json.loads((args.run / "config.json").read_text())["config"]
    config = RunConfig.model_validate(saved)
    embedder = make_embedder(config.embed, Path(config.cache_dir))
    result = calibrate_run(args.run, load_truths(args.truth), embedder, config)
    print(f"Embedding model: {embedder.model_id}")
    print(f"dedup threshold: {result['dedup'].chosen}" + ("" if result["dedup"].cap_met else " (false-merge cap not met; chosen for best recall/precision balance)"))
    print(f"novelty threshold: {result['novelty'].chosen}")
    (args.run / "eval").mkdir(exist_ok=True)
    (args.run / "eval" / "calibration.json").write_text(
        json.dumps({"model": embedder.model_id, **{k: {"chosen": v.chosen, "cap_met": v.cap_met, "rows": v.rows} for k, v in result.items()}}, indent=2)
    )
    if args.write:
        data = json.loads(args.write.read_text())
        data.setdefault("dedup", {})["threshold"] = result["dedup"].chosen
        data.setdefault("novelty", {})["threshold"] = result["novelty"].chosen
        args.write.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Updated {args.write}")
    return 0


def _add_report(sub: Any) -> None:
    parser = sub.add_parser("report", help="Write report.md for a run")
    parser.add_argument("--run", type=Path, required=True)
    parser.set_defaults(func=_cmd_report)


def _cmd_report(args: argparse.Namespace) -> int:
    from goldenset.evaluation.report import write_report

    print(write_report(args.run))
    return 0


# --- review / finalize ---


def _add_review(sub: Any) -> None:
    parser = sub.add_parser("review", help="Open the Streamlit review page for a run")
    parser.add_argument("--run", type=Path, required=True)
    parser.set_defaults(func=_cmd_review)


def _cmd_review(args: argparse.Namespace) -> int:
    app = Path(__file__).resolve().parent / "review" / "app.py"
    env = {**os.environ, "GOLDENSET_RUN_DIR": str(args.run.resolve())}
    return subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], env=env, check=False).returncode


def _add_finalize(sub: Any) -> None:
    parser = sub.add_parser("finalize", help="Build golden_set.jsonl from reviewer decisions")
    parser.add_argument("--run", type=Path, required=True)
    parser.set_defaults(func=_cmd_finalize)


def _cmd_finalize(args: argparse.Namespace) -> int:
    from goldenset.evaluation.report import write_report
    from goldenset.review.finalize import write_golden

    result = write_golden(args.run)
    print(f"Wrote {len(result.records)} golden records to {args.run / 'golden_set.jsonl'}")
    if result.skipped:
        print(f"Skipped {len(result.skipped)} accepted records the model could not label: {', '.join(result.skipped)}")
    if (args.run / "stats.json").exists():
        write_report(args.run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goldenset", description="Golden-set pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for add in (_add_generate, _add_make_seed, _add_run, _add_eval, _add_calibrate, _add_report, _add_review, _add_finalize):
        add(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
