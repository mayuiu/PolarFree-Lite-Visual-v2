import argparse
import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


IMAGE_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ("_raw_result", "raw_result"),
    ("_comparison", "comparison"),
    ("_mask", "mask"),
    ("_result", "result"),
)

METRIC_KEYS: Tuple[str, ...] = (
    "raw_focus_l1",
    "focus_l1",
    "visual_focus_l1",
    "raw_l1",
    "l1",
    "visual_l1",
)


@dataclass
class SampleBundle:
    name: str
    files: Dict[str, Path] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    category: str = "likely_medium"
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML/Markdown index for PolarFree test_results.")
    parser.add_argument(
        "test_results_dir",
        type=Path,
        help=(
            "Directory containing all_summary.json and images/, or a parent output directory "
            "containing the latest */test_results folder."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path. Defaults to <test_results_dir>/analysis_report.html or .md.",
    )
    parser.add_argument("--format", choices=("html", "md"), default="html")
    parser.add_argument(
        "--max-thumbnails",
        type=int,
        default=0,
        help="Maximum thumbnails per category in HTML. 0 means include all.",
    )
    return parser.parse_args()


def resolve_test_results_dir(path: Path) -> Path:
    if (path / "all_summary.json").exists():
        return path

    direct_child = path / "test_results"
    if (direct_child / "all_summary.json").exists():
        return direct_child

    candidates = sorted(
        (candidate.parent for candidate in path.glob("*/test_results/all_summary.json")),
        key=lambda item: (item / "all_summary.json").stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Unable to find all_summary.json in {path}, {direct_child}, or one-level */test_results directories."
    )


def read_summary(test_results_dir: Path) -> Dict[str, object]:
    summary_path = test_results_dir / "all_summary.json"
    if not summary_path.exists():
        return {"_warning": f"Missing summary file: {summary_path}"}
    with summary_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def split_sample_name(path: Path) -> Optional[Tuple[str, str]]:
    stem = path.stem
    for suffix, kind in IMAGE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], kind
    return None


def collect_samples(images_dir: Path) -> Dict[str, SampleBundle]:
    samples: Dict[str, SampleBundle] = {}
    if not images_dir.exists():
        return samples
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = split_sample_name(path)
        if parsed is None:
            continue
        sample_name, kind = parsed
        bundle = samples.setdefault(sample_name, SampleBundle(name=sample_name))
        bundle.files[kind] = path
    return samples


def iter_metric_records(summary: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    for key in ("sample_metrics", "per_sample", "records", "items", "results", "sample_outputs"):
        value = summary.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    yield item


def sample_name_from_record(record: Mapping[str, object]) -> Optional[str]:
    for key in ("sample", "sample_name", "name", "prefix", "id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return Path(value).stem

    scene = record.get("scene")
    group = record.get("group")
    if isinstance(scene, str) and isinstance(group, str):
        return f"{scene}_{group}"

    for key in ("comparison_path", "raw_result_path", "result_path", "mask_path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            parsed = split_sample_name(Path(value))
            if parsed is not None:
                return parsed[0]
    return None


def attach_metrics(samples: Dict[str, SampleBundle], summary: Mapping[str, object]) -> bool:
    found = False
    for record in iter_metric_records(summary):
        sample_name = sample_name_from_record(record)
        if not sample_name or sample_name not in samples:
            continue
        for key in METRIC_KEYS:
            value = record.get(key)
            if isinstance(value, (int, float)):
                samples[sample_name].metrics[key] = float(value)
                found = True
    return found


def metric_for_category(sample: SampleBundle) -> Optional[float]:
    for key in METRIC_KEYS:
        if key in sample.metrics:
            return sample.metrics[key]
    return None


def quantiles(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    low_index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.33))))
    high_index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.66))))
    return ordered[low_index], ordered[high_index]


def categorize_samples(samples: Dict[str, SampleBundle], has_sample_metrics: bool) -> None:
    if has_sample_metrics:
        values = [value for sample in samples.values() if (value := metric_for_category(sample)) is not None]
        good_cutoff, bad_cutoff = quantiles(values)
        for sample in samples.values():
            value = metric_for_category(sample)
            if value is None:
                sample.category = "likely_medium"
                sample.reason = "No per-sample metric found."
            elif value <= good_cutoff:
                sample.category = "likely_good"
                sample.reason = f"Per-sample focus/L1 metric is low: {value:.6f}."
            elif value >= bad_cutoff:
                sample.category = "likely_bad"
                sample.reason = f"Per-sample focus/L1 metric is high: {value:.6f}."
            else:
                sample.category = "likely_medium"
                sample.reason = f"Per-sample focus/L1 metric is mid-range: {value:.6f}."
        return

    for sample in samples.values():
        has_comparison = "comparison" in sample.files
        has_raw = "raw_result" in sample.files
        has_result = "result" in sample.files
        has_mask = "mask" in sample.files
        if has_comparison and has_raw and has_result and has_mask:
            sample.category = "likely_good"
            sample.reason = "Complete file bundle; no per-sample metrics were available."
        elif has_comparison and (has_raw or has_result):
            sample.category = "likely_medium"
            sample.reason = "Partial file bundle; no per-sample metrics were available."
        else:
            sample.category = "likely_bad"
            sample.reason = "Missing comparison or output images."


def grouped_samples(samples: Mapping[str, SampleBundle]) -> Dict[str, List[SampleBundle]]:
    groups = {"likely_good": [], "likely_medium": [], "likely_bad": []}
    for sample in sorted(samples.values(), key=lambda item: item.name):
        groups.setdefault(sample.category, []).append(sample)
    return groups


def relative_link(target: Path, report_path: Path) -> str:
    try:
        return target.resolve().relative_to(report_path.parent.resolve()).as_posix()
    except ValueError:
        return target.resolve().as_posix()


def summary_rows(summary: Mapping[str, object]) -> List[Tuple[str, str]]:
    keys = (
        "split",
        "count",
        "render_mode",
        "img_size",
        "raw_l1",
        "raw_focus_l1",
        "raw_psnr",
        "raw_ssim",
        "visual_l1",
        "visual_focus_l1",
        "visual_psnr",
        "visual_ssim",
        "checkpoint_epoch",
    )
    rows: List[Tuple[str, str]] = []
    for key in keys:
        if key in summary:
            value = summary[key]
            rows.append((key, f"{value:.6g}" if isinstance(value, float) else str(value)))
    if "_warning" in summary:
        rows.append(("warning", str(summary["_warning"])))
    return rows


def raw_visual_note(summary: Mapping[str, object]) -> str:
    raw_focus = summary.get("raw_focus_l1")
    visual_focus = summary.get("visual_focus_l1")
    raw_l1 = summary.get("raw_l1")
    visual_l1 = summary.get("visual_l1")
    notes: List[str] = []
    if isinstance(raw_focus, (int, float)) and isinstance(visual_focus, (int, float)):
        winner = "raw" if raw_focus <= visual_focus else "visual"
        notes.append(f"focus_l1 winner: {winner} ({raw_focus:.4f} vs {visual_focus:.4f}).")
    if isinstance(raw_l1, (int, float)) and isinstance(visual_l1, (int, float)):
        winner = "raw" if raw_l1 <= visual_l1 else "visual"
        notes.append(f"l1 winner: {winner} ({raw_l1:.4f} vs {visual_l1:.4f}).")
    return " ".join(notes) if notes else "No raw/visual comparison metrics found in all_summary.json."


def render_html(
    report_path: Path,
    test_results_dir: Path,
    summary: Mapping[str, object],
    samples: Mapping[str, SampleBundle],
    has_sample_metrics: bool,
    max_thumbnails: int,
) -> str:
    groups = grouped_samples(samples)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = "\n".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>" for key, value in summary_rows(summary)
    )
    category_counts = " | ".join(f"{name}: {len(items)}" for name, items in groups.items())
    basis = "per-sample metrics" if has_sample_metrics else "file completeness only"

    sections: List[str] = []
    for category, items in groups.items():
        shown = items if max_thumbnails <= 0 else items[:max_thumbnails]
        cards: List[str] = []
        for sample in shown:
            comparison = sample.files.get("comparison")
            image_html = ""
            if comparison is not None:
                image_src = html.escape(relative_link(comparison, report_path))
                image_html = f'<a href="{image_src}"><img src="{image_src}" alt="{html.escape(sample.name)}"></a>'
            links = []
            for kind in ("comparison", "raw_result", "result", "mask"):
                path = sample.files.get(kind)
                if path is not None:
                    href = html.escape(relative_link(path, report_path))
                    links.append(f'<a href="{href}">{kind}</a>')
                else:
                    links.append(f'<span class="missing">{kind}</span>')
            metric_text = ", ".join(f"{key}={value:.6f}" for key, value in sorted(sample.metrics.items()))
            if not metric_text:
                metric_text = "no per-sample metric"
            cards.append(
                "<article class=\"card\">"
                f"<h3>{html.escape(sample.name)}</h3>"
                f"{image_html}"
                f"<p>{' | '.join(links)}</p>"
                f"<p class=\"reason\">{html.escape(sample.reason)}</p>"
                f"<p class=\"metric\">{html.escape(metric_text)}</p>"
                "</article>"
            )
        more = ""
        if max_thumbnails > 0 and len(items) > max_thumbnails:
            more = f"<p>Showing {max_thumbnails} of {len(items)} samples.</p>"
        sections.append(
            f"<section><h2>{html.escape(category)} ({len(items)})</h2>{more}<div class=\"grid\">{''.join(cards)}</div></section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PolarFree Test Results Analysis</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2, h3 {{ margin: 0.4em 0; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d6d9de; padding: 6px 10px; text-align: left; }}
    th {{ background: #f2f4f7; }}
    .note {{ background: #fff7d6; border: 1px solid #e5ca66; padding: 10px 12px; margin: 12px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #d6d9de; padding: 10px; border-radius: 6px; background: #fff; }}
    .card img {{ width: 100%; height: auto; border: 1px solid #eceff3; }}
    .missing {{ color: #a33; text-decoration: line-through; }}
    .reason, .metric {{ color: #52606d; font-size: 0.92em; }}
  </style>
</head>
<body>
  <h1>PolarFree Test Results Analysis</h1>
  <p>Generated at {html.escape(generated_at)}</p>
  <p>Test results dir: <code>{html.escape(str(test_results_dir))}</code></p>
  <div class="note">
    <strong>Category basis:</strong> {html.escape(basis)}. {html.escape(raw_visual_note(summary))}
  </div>
  <h2>Summary</h2>
  <table>{rows}</table>
  <p>{html.escape(category_counts)}</p>
  {''.join(sections)}
</body>
</html>
"""


def render_markdown(
    report_path: Path,
    test_results_dir: Path,
    summary: Mapping[str, object],
    samples: Mapping[str, SampleBundle],
    has_sample_metrics: bool,
) -> str:
    groups = grouped_samples(samples)
    lines = [
        "# PolarFree Test Results Analysis",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Test results dir: `{test_results_dir}`",
        "",
        f"Category basis: {'per-sample metrics' if has_sample_metrics else 'file completeness only'}.",
        "",
        raw_visual_note(summary),
        "",
        "## Summary",
        "",
        "| key | value |",
        "|---|---|",
    ]
    for key, value in summary_rows(summary):
        lines.append(f"| `{key}` | `{value}` |")
    for category, items in groups.items():
        lines.extend(["", f"## {category} ({len(items)})", ""])
        for sample in items:
            comparison = sample.files.get("comparison")
            if comparison is not None:
                link = relative_link(comparison, report_path)
                lines.append(f"- [{sample.name}]({link}) - {sample.reason}")
            else:
                lines.append(f"- {sample.name} - {sample.reason}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    test_results_dir = resolve_test_results_dir(args.test_results_dir)
    output_path = args.output
    if output_path is None:
        output_path = test_results_dir / f"analysis_report.{args.format}"

    summary = read_summary(test_results_dir)
    images_dir = test_results_dir / "images"
    samples = collect_samples(images_dir)
    has_sample_metrics = attach_metrics(samples, summary)
    categorize_samples(samples, has_sample_metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "html":
        content = render_html(output_path, test_results_dir, summary, samples, has_sample_metrics, args.max_thumbnails)
    else:
        content = render_markdown(output_path, test_results_dir, summary, samples, has_sample_metrics)
    output_path.write_text(content, encoding="utf-8")

    groups = grouped_samples(samples)
    print(f"summary={test_results_dir / 'all_summary.json'}")
    print(f"images={images_dir}")
    print(f"samples={len(samples)}")
    for category, items in groups.items():
        print(f"{category}={len(items)}")
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
