import csv
import json
import re
from pathlib import Path
from statistics import mean


# Edit this block before each experiment, then run:
#   python fdong/scripts/extract_train_log.py
EXPERIMENT_NAME = "unet-4-8-16-8-4"
LOG_FILE = f"../logs/{EXPERIMENT_NAME}.log"
OUTPUT_DIR = f"../experiments/{EXPERIMENT_NAME}"
RETRIEVED_LINES = 100
TAIL_LINES = 100

CONFIG_HEADER = "Training Configuration:"
CONFIG_RE = re.compile(r"^\s{2,}([A-Za-z0-9_]+):\s*(.*)\s*$")
TRAIN_RE = re.compile(
    r"batch:\s*(?P<global_step>-?\d+)-(?P<local_batch>\d+),\s*"
    r"loss:\s*(?P<loss>[-+0-9.eE]+),\s*"
    r"batch_time:\s*(?P<batch_time>[-+0-9.eE]+)"
)


def parse_log(log_path):
    config = {}
    records = []
    in_config = False

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, 1):
        stripped = line.rstrip("\n")

        if stripped.strip() == CONFIG_HEADER:
            in_config = True
            continue

        if in_config:
            match = CONFIG_RE.match(stripped)
            if match:
                key, value = match.group(1), match.group(2)
                config[key] = value
                continue
            if stripped.strip() == "":
                continue
            in_config = False

        match = TRAIN_RE.search(stripped)
        if match:
            records.append(
                {
                    "line_no": line_no,
                    "global_step": int(match.group("global_step")),
                    "local_batch": int(match.group("local_batch")),
                    "loss": float(match.group("loss")),
                    "batch_time": float(match.group("batch_time")),
                }
            )

    return config, records, lines


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sample_records(records, every):
    if every <= 1:
        return records
    return [record for idx, record in enumerate(records) if idx % every == 0]


def aggregate_records(records, window):
    if window <= 1:
        return [
            {
                "start_record": idx,
                "end_record": idx,
                "start_global_step": record["global_step"],
                "end_global_step": record["global_step"],
                "loss_mean": record["loss"],
                "loss_min": record["loss"],
                "loss_max": record["loss"],
                "batch_time_mean": record["batch_time"],
                "num_records": 1,
            }
            for idx, record in enumerate(records)
        ]

    aggregated = []
    for start in range(0, len(records), window):
        chunk = records[start : start + window]
        losses = [record["loss"] for record in chunk]
        batch_times = [record["batch_time"] for record in chunk]
        aggregated.append(
            {
                "start_record": start,
                "end_record": start + len(chunk) - 1,
                "start_global_step": chunk[0]["global_step"],
                "end_global_step": chunk[-1]["global_step"],
                "loss_mean": mean(losses),
                "loss_min": min(losses),
                "loss_max": max(losses),
                "batch_time_mean": mean(batch_times),
                "num_records": len(chunk),
            }
        )
    return aggregated


def build_summary(experiment_name, log_path, config, records):
    summary = {
        "experiment_name": experiment_name,
        "log_path": str(log_path),
        "num_records": len(records),
        "config": config,
    }

    if records:
        losses = [record["loss"] for record in records]
        batch_times = [record["batch_time"] for record in records]
        summary.update(
            {
                "first_record": records[0],
                "last_record": records[-1],
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_min": min(losses),
                "loss_max": max(losses),
                "loss_mean": mean(losses),
                "batch_time_mean": mean(batch_times),
            }
        )
    return summary


def write_tail(path, lines, tail_lines):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines[-tail_lines:])


def main():
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        raise FileNotFoundError(
            f"LOG_FILE does not exist: {log_path}\n"
            "Edit LOG_FILE near the top of this script before running it."
        )

    experiment_name = EXPERIMENT_NAME or log_path.stem
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path("experiments") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config, records, lines = parse_log(log_path)
    sample_every = len(records) // RETRIEVED_LINES
    aggregate_window = len(records) // RETRIEVED_LINES
    sampled = sample_records(records, sample_every)
    aggregated = aggregate_records(records, aggregate_window)
    summary = build_summary(experiment_name, log_path, config, records)
    summary.update(
        {
            "sample_every": sample_every,
            "aggregate_window": aggregate_window,
            "tail_lines": TAIL_LINES,
            "outputs": {
                "summary": "summary.json",
                "sampled_records": "loss_sampled.csv",
                "aggregated_records": "loss_aggregated.csv",
                "tail": "train_tail.log",
            },
        }
    )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_csv(
        output_dir / "loss_sampled.csv",
        sampled,
        ["line_no", "global_step", "local_batch", "loss", "batch_time"],
    )
    write_csv(
        output_dir / "loss_aggregated.csv",
        aggregated,
        [
            "start_record",
            "end_record",
            "start_global_step",
            "end_global_step",
            "loss_mean",
            "loss_min",
            "loss_max",
            "batch_time_mean",
            "num_records",
        ],
    )
    write_tail(output_dir / "train_tail.log", lines, TAIL_LINES)

    print(f"Parsed {len(records)} training records from {log_path}")
    print(f"Wrote summary to {output_dir}")
    if records:
        print(f"loss: first={records[0]['loss']:.4f}, last={records[-1]['loss']:.4f}, min={summary['loss_min']:.4f}")


if __name__ == "__main__":
    main()
