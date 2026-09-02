import argparse
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Optional

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.lvbench.utils import (
    add_common_arguments,
    aggregate_task_metrics,
    append_record,
    attach_task_metrics,
    build_agent_config,
    create_run_dir,
    ensure_run_manifest,
    get_agent_class,
    get_agent_model_name,
    get_agent_tools,
    is_completed_record,
    load_latest_records,
    safe_config,
    tee_output,
    write_json,
    write_trajectory,
)


DEFAULT_METADATA_PATH = "./data/LVBench/zai-org-LVBench/video_info.meta.jsonl"
DEFAULT_VIDEO_DIR = "./data/LVBench/AIWinter-LVBench/all_videos"
SAMPLE_MANIFEST_PATH = Path(__file__).with_name("samples.json")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a video Agent on LVBench.")
    add_common_arguments(parser)
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    return parser.parse_args(argv)


def load_dataset(metadata_path: Path) -> list[dict]:
    samples = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for video in map(json.loads, file):
            for question in video["qa"]:
                samples.append(
                    {
                        "uid": str(question["uid"]),
                        "video_id": video["key"],
                        "question": question["question"],
                        "answer": question["answer"],
                        "question_types": question["question_type"],
                    }
                )
    return samples


def select_samples(samples: list[dict], mode: str) -> list[dict]:
    if mode == "full":
        return samples
    manifest = json.loads(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    selected_uids = set(manifest[mode])
    selected = [sample for sample in samples if sample["uid"] in selected_uids]
    if len(selected) != len(selected_uids):
        found = {sample["uid"] for sample in selected}
        raise ValueError(f"Unknown LVBench sample IDs: {sorted(selected_uids - found)}")
    return selected


def build_question(question: str) -> str:
    return (
        f"{question}\n\n"
        "Output requirement: Return exactly one uppercase option letter: A, B, "
        "C, or D. Do not include parentheses, explanation, or any other text."
    )


def normalize_answer(answer: str) -> str:
    text = answer.strip().upper()
    direct = re.fullmatch(r"\s*\(?([A-D])\)?[\s.:,\-]*", text)
    if direct:
        return direct.group(1)

    strong_candidates = []
    labelled_patterns = (
        r"\b(?:FINAL|BEST|CORRECT)\s+"
        r"(?:ANSWER|OPTION|CHOICE)(?:'S\s+LETTER)?\s*"
        r"(?:IS\s*:?|[:=])?\s*\**\s*\(?\s*([A-D])\s*\)?\s*\**",
        r"\b(?:ANSWER|OPTION|CHOICE)(?:'S\s+LETTER)?\s*"
        r"(?:IS\s*:?|[:=])\s*\**\s*\(?\s*([A-D])\s*\)?\s*\**",
    )
    for pattern in labelled_patterns:
        strong_candidates.extend(
            (match.start(1), match.group(1))
            for match in re.finditer(pattern, text)
        )

    last_line = next(
        (line for line in reversed(text.splitlines()) if line.strip()), ""
    )
    line_offset = text.rfind(last_line)
    formatted_line = re.fullmatch(
        r"\s*\**\s*(?:"
        r"\(\s*([A-D])\s*\)|"
        r"\[\s*([A-D])\s*\]|"
        r"([A-D])\s*[.):\-]"
        r")\s*\**(?:\s+.+)?\s*",
        last_line,
    )
    if formatted_line:
        option = next(group for group in formatted_line.groups() if group)
        strong_candidates.append((line_offset, option))
    else:
        bare_line = re.fullmatch(r"\s*\**\s*([A-D])\s*\**[.,:]?\s*", last_line)
        if bare_line:
            strong_candidates.append((line_offset, bare_line.group(1)))

    if strong_candidates:
        return max(strong_candidates, key=lambda candidate: candidate[0])[1]

    candidates = set(re.findall(r"\b([A-D])\b", text))
    return next(iter(candidates)) if len(candidates) == 1 else ""


def compute_metrics(samples: list[dict], records: dict[str, dict]) -> dict:
    totals = defaultdict(int)
    correct = defaultdict(int)
    overall_correct = 0
    for sample in samples:
        prediction = records.get(sample["uid"], {}).get("prediction", "")
        is_correct = prediction == sample["answer"]
        overall_correct += int(is_correct)
        for question_type in sample["question_types"]:
            totals[question_type] += 1
            correct[question_type] += int(is_correct)
    count = len(samples)
    return {
        "overall": {
            "accuracy": overall_correct / count if count else 0.0,
            "correct": overall_correct,
            "total": count,
        },
        "by_question_type": {
            question_type: {
                "accuracy": correct[question_type] / total,
                "correct": correct[question_type],
                "total": total,
            }
            for question_type, total in sorted(totals.items())
        },
    }


def write_markdown_table(
    path: Path, columns: list[str], rows: list[list[object]]
) -> None:
    def escape(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "# LVBench Run Report",
        "",
        f"| {' | '.join(map(escape, columns))} |",
        f"| {' | '.join('---' for _ in columns)} |",
    ]
    lines.extend(f"| {' | '.join(map(escape, row))} |" for row in rows)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def write_report(
    run_dir: Path,
    agent_name: str,
    mode: str,
    model_name: str,
    samples: list[dict],
    records: dict[str, dict],
    metrics: dict,
) -> None:
    def make_row(
        task: str, task_samples: list[dict], accuracy: float
    ) -> list[object]:
        keys = [sample["uid"] for sample in task_samples]
        statistics = aggregate_task_metrics(keys, records)
        completed = sum(
            records.get(key, {}).get("status") == "success" for key in keys
        )
        return [
            task,
            agent_name,
            mode,
            model_name,
            f"{completed}/{len(keys)}",
            f"{accuracy:.4f}",
            f"{statistics['agent_rounds']['average']:.2f}",
            statistics["token_usage"]["total_tokens"],
            f"{statistics['elapsed_seconds']['total']:.2f}",
        ]

    rows = [make_row("Overall", samples, metrics["overall"]["accuracy"])]
    for task, task_metrics in metrics["by_question_type"].items():
        task_samples = [
            sample for sample in samples if task in sample["question_types"]
        ]
        rows.append(make_row(task, task_samples, task_metrics["accuracy"]))

    write_markdown_table(
        run_dir / "report.md",
        [
            "Task",
            "Agent",
            "Mode",
            "Model",
            "Completed",
            "Accuracy",
            "Avg. rounds",
            "Tokens",
            "Time (s)",
        ],
        rows,
    )


def run_benchmark(args: argparse.Namespace, agent_class=None) -> int:
    metadata_path = Path(args.metadata_path).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve()
    samples = select_samples(load_dataset(metadata_path), args.mode)
    selected_keys = [sample["uid"] for sample in samples]
    agent_name = getattr(args, "agent", "videoseek")
    config = build_agent_config(args)
    model_name = get_agent_model_name(config, agent_name)
    run_dir = create_run_dir(
        "lvbench", args.mode, args.output_dir, args.run_dir, agent_name, model_name
    )
    progress_stream = sys.stderr
    with tee_output(run_dir / "run.log"):
        print(
            f"Starting LVBench: agent={agent_name}, mode={args.mode}, "
            f"samples={len(samples)}, "
            f"verbose={args.verbose}",
            flush=True,
        )
        print(f"Log file: {run_dir / 'run.log'}", flush=True)
        try:
            return _execute_benchmark(
                args=args,
                agent_name=agent_name,
                agent_class=agent_class,
                config=config,
                metadata_path=metadata_path,
                progress_stream=progress_stream,
                run_dir=run_dir,
                samples=samples,
                selected_keys=selected_keys,
                video_dir=video_dir,
            )
        except Exception:
            traceback.print_exc()
            raise


def _execute_benchmark(
    args: argparse.Namespace,
    agent_name: str,
    agent_class,
    config: dict,
    metadata_path: Path,
    progress_stream,
    run_dir: Path,
    samples: list[dict],
    selected_keys: list[str],
    video_dir: Path,
) -> int:
    ensure_run_manifest(
        run_dir,
        {
            "dataset": "lvbench",
            "agent": agent_name,
            "mode": args.mode,
            "metadata_path": str(metadata_path),
            "video_dir": str(video_dir),
            "sample_keys": selected_keys,
            "agent_config": safe_config(config),
        },
    )

    if agent_class is None:
        agent_class = get_agent_class(agent_name)

    records_path = run_dir / "records.jsonl"
    latest = load_latest_records(records_path)
    agent = None
    current_video_id = None

    progress = tqdm(
        total=len(samples),
        desc=f"LVBench {args.mode}",
        dynamic_ncols=True,
        file=progress_stream,
        unit="question",
    )
    for sample in samples:
        uid = sample["uid"]
        if is_completed_record(latest.get(uid, {})):
            progress.update(1)
            progress.set_postfix_str(f"uid={uid} skipped")
            continue
        task_agent = None
        trajectory_dict = None
        started_at = perf_counter()
        try:
            video_path = video_dir / f"{sample['video_id']}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"Video not found: {video_path}")
            if sample["video_id"] != current_video_id:
                agent = agent_class(
                    config=config,
                    video_path=str(video_path),
                    subtitle_path=None,
                    output_dir=str(run_dir),
                    tools=get_agent_tools(config, agent_name),
                    verbose=args.verbose,
                )
                current_video_id = sample["video_id"]
            task_agent = agent
            trajectory = agent.run(build_question(sample["question"]))
            trajectory_dict = trajectory.to_dict()
            raw_prediction = trajectory_dict.get("final_answer", "")
            record = {
                "sample_key": uid,
                "status": "success",
                "video_id": sample["video_id"],
                "question_types": sample["question_types"],
                "raw_prediction": raw_prediction,
                "prediction": normalize_answer(raw_prediction),
                "answer": sample["answer"],
                "trajectory": write_trajectory(run_dir, uid, trajectory_dict),
            }
        except Exception as error:
            traceback.print_exc()
            record = {
                "sample_key": uid,
                "status": "error",
                "video_id": sample["video_id"],
                "error": f"{type(error).__name__}: {error}",
            }
        attach_task_metrics(
            record,
            task_agent,
            trajectory_dict,
            perf_counter() - started_at,
        )
        append_record(records_path, record)
        latest[uid] = record
        progress.update(1)
        progress.set_postfix_str(
            f"uid={uid} {record['status']} "
            f"rounds={record['agent_rounds']} "
            f"tokens={record['token_usage']['total_tokens']} "
            f"time={record['elapsed_seconds']:.1f}s"
        )
        print(
            f"Sample {uid}: {record['status']}, "
            f"rounds={record['agent_rounds']}, "
            f"tokens={record['token_usage']['total_tokens']}, "
            f"elapsed={record['elapsed_seconds']:.3f}s"
        )
    progress.close()

    predictions = {
        uid: latest.get(uid, {}).get("prediction", "") for uid in selected_keys
    }
    write_json(run_dir / "predictions.json", predictions)
    metrics = compute_metrics(samples, latest)
    metrics["run_statistics"] = aggregate_task_metrics(selected_keys, latest)
    write_json(run_dir / "metrics.json", metrics)
    failures = sum(latest.get(uid, {}).get("status") != "success" for uid in selected_keys)
    write_report(
        run_dir,
        agent_name,
        args.mode,
        get_agent_model_name(config, agent_name),
        samples,
        latest,
        metrics,
    )
    print(f"Report: {run_dir / 'report.md'}")
    print(f"Run directory: {run_dir}")
    print(f"Completed: {len(samples) - failures}/{len(samples)}, failures: {failures}")
    return 1 if failures else 0


def main(argv: Optional[list[str]] = None) -> int:
    return run_benchmark(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
