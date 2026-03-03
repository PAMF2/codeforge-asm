from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path


def pct(v: float) -> str:
    return f"{100.0 * float(v):.1f}%"


def f3(v: float) -> str:
    return f"{float(v):.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a paper-friendly HTML report from Assembly-SWE aggregate.json")
    p.add_argument("--aggregate", required=True, help="Path to aggregate.json")
    p.add_argument("--out", default="", help="Output HTML path (default: same dir/report.html)")
    p.add_argument("--title", default="Assembly-SWE Evaluation Report")
    p.add_argument("--curves", default="", help="Optional curves png path to embed")
    args = p.parse_args()

    aggregate = Path(args.aggregate).resolve()
    if not aggregate.exists():
        raise FileNotFoundError(f"aggregate not found: {aggregate}")
    out = Path(args.out).resolve() if args.out else aggregate.parent / "report.html"
    curves = Path(args.curves).resolve() if args.curves else aggregate.parent / "curves.png"

    data = json.loads(aggregate.read_text(encoding="utf-8"))
    rows = sorted(data.get("rows", []), key=lambda x: x.get("iter", 0))
    if not rows:
        raise RuntimeError("No rows found in aggregate.json")

    best_correct = max(rows, key=lambda x: (x.get("correct_rate_at_1", -1), x.get("avg_reward_at_1", -1)))
    best_assembly = max(rows, key=lambda x: (x.get("assembly_rate_at_1", -1), x.get("avg_reward_at_1", -1)))
    best_reward = max(rows, key=lambda x: x.get("avg_reward_at_1", -1))
    top5 = sorted(rows, key=lambda x: (x.get("correct_rate_at_1", -1), x.get("avg_reward_at_1", -1)), reverse=True)[:5]

    image_html = "<p><i>curves.png not found</i></p>"
    if curves.exists():
        image_b64 = base64.b64encode(curves.read_bytes()).decode("utf-8")
        image_html = (
            f'<img src="data:image/png;base64,{image_b64}" alt="curves" '
            'style="max-width:100%;border:1px solid #ddd;border-radius:8px;" />'
        )

    rows_html = "\n".join(
        (
            "<tr>"
            f"<td>{r.get('iter')}</td>"
            f"<td>{pct(r.get('correct_rate_at_1', 0.0))}</td>"
            f"<td>{pct(r.get('assembly_rate_at_1', 0.0))}</td>"
            f"<td>{f3(r.get('avg_reward_at_1', 0.0))}</td>"
            f"<td>{f3(r.get('pass_at', {}).get('1', 0.0))}</td>"
            f"<td>{f3(r.get('pass_at', {}).get('3', 0.0))}</td>"
            f"<td>{f3(r.get('pass_at', {}).get('5', 0.0))}</td>"
            "</tr>"
        )
        for r in rows
    )
    top5_html = "\n".join(
        (
            f"<li><b>iter {r.get('iter')}</b> | "
            f"correct@1={pct(r.get('correct_rate_at_1', 0.0))}, "
            f"assembly@1={pct(r.get('assembly_rate_at_1', 0.0))}, "
            f"avg_reward@1={f3(r.get('avg_reward_at_1', 0.0))}</li>"
        )
        for r in top5
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{args.title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    .card {{ border:1px solid #ddd; border-radius:10px; padding:14px; margin:12px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
    th {{ background: #f5f5f5; }}
    code {{ background:#f3f3f3; padding:2px 6px; border-radius:6px; }}
  </style>
</head>
<body>
  <h1>{args.title}</h1>
  <p>Generated at {now}</p>

  <div class="card">
    <h2>Benchmark</h2>
    <p><b>Tasks:</b> <code>{data.get("tasks","")}</code></p>
    <p><b>Iterations:</b> {data.get("iter_start")}..{data.get("iter_end")} | evaluated={data.get("count_evaluated")} | skipped={data.get("count_skipped", 0)}</p>
    <ul>
      <li><b>correct_rate@1</b>: first-try correctness rate (primary).</li>
      <li><b>assembly_rate@1</b>: first-try assembly/compile rate.</li>
      <li><b>pass@k</b>: success if any of top-k candidates is correct.</li>
      <li><b>avg_reward@1</b>: average evaluator reward for top-1 candidate.</li>
    </ul>
  </div>

  <div class="card">
    <h2>Highlights</h2>
    <ul>
      <li>Best correct@1: iter {best_correct.get("iter")} ({pct(best_correct.get("correct_rate_at_1", 0.0))})</li>
      <li>Best assembly@1: iter {best_assembly.get("iter")} ({pct(best_assembly.get("assembly_rate_at_1", 0.0))})</li>
      <li>Best avg_reward@1: iter {best_reward.get("iter")} ({f3(best_reward.get("avg_reward_at_1", 0.0))})</li>
    </ul>
    <h3>Top-5 checkpoints</h3>
    <ol>{top5_html}</ol>
  </div>

  <div class="card">
    <h2>Curves</h2>
    {image_html}
  </div>

  <div class="card">
    <h2>Per-iteration table</h2>
    <table>
      <thead>
        <tr>
          <th>iter</th>
          <th>correct@1</th>
          <th>assembly@1</th>
          <th>avg_reward@1</th>
          <th>pass@1</th>
          <th>pass@3</th>
          <th>pass@5</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    print(f"[report_html] wrote: {out}")


if __name__ == "__main__":
    main()
