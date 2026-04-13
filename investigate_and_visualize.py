#!/usr/bin/env python3
"""Fetch, investigate, and visualize a news digest in one command.

Examples:
    python3 investigate_and_visualize.py \
        --query "important biotech and oncology developments this week" \
        --output reports/latest.json \
        --html-output web/latest.html

    python3 investigate_and_visualize.py \
        --query "AI drug discovery funding and partnerships" \
        --markdown-output reports/ai-drug-discovery.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

from demo_fetch import (
    DEFAULT_CONFIG_PATH,
    annotate_items,
    batched,
    build_report_payload,
    classify_batch,
    gather_items,
    render_markdown,
    resolve_settings,
)
from visualize_report import normalize_report, render_html


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "news-digest"


def derive_paths(
    *,
    query: str,
    json_output: str | None,
    markdown_output: str | None,
    html_output: str | None,
) -> tuple[Path, Path | None, Path]:
    today = dt.datetime.now().strftime("%Y-%m-%d")
    default_stem = f"{today}-{slugify(query)}"

    json_path = Path(json_output) if json_output else Path("reports") / f"{default_stem}.json"
    markdown_path = Path(markdown_output) if markdown_output else None

    if html_output:
        html_path = Path(html_output)
    elif json_path.parent.name == "reports":
        html_path = json_path.parent.parent / "web" / f"{json_path.stem}.html"
    else:
        html_path = json_path.with_suffix(".html")

    return json_path, markdown_path, html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to TOML config file")
    parser.add_argument("--query", required=True, help="What kind of news the user wants")
    parser.add_argument("--output", help="Write the JSON report to this path")
    parser.add_argument("--markdown-output", help="Optional markdown export path")
    parser.add_argument("--html-output", help="Write the HTML visualization to this path")
    parser.add_argument("--days", type=int, help="How many days back to scan")
    parser.add_argument("--batch-size", type=int, help="Articles per LLM call")
    parser.add_argument("--max-items", type=int, help="Max candidate articles to review")
    parser.add_argument("--top-k", type=int, help="How many matching articles to keep")
    parser.add_argument("--min-relevance", type=int, help="Minimum relevance to keep")
    parser.add_argument("--model", help="OpenAI-compatible chat model")
    parser.add_argument("--base-url", help="Chat completion base URL")
    parser.add_argument("--api-key", help="API key for the chat endpoint")
    parser.add_argument("--timeout", type=int, help="HTTP timeout in seconds")
    return parser.parse_args()


def investigate(settings: dict[str, Any]) -> dict[str, Any]:
    items = gather_items(settings["days"], settings["feeds"])
    if not items:
        return build_report_payload(
            query=settings["query"],
            kept=[],
            total_items=0,
            days=settings["days"],
            model=settings["model"],
        )

    items = items[: settings["max_items"]]
    annotate_items(items)

    verdicts: dict[str, dict[str, Any]] = {}
    for batch in batched(items, settings["batch_size"]):
        verdicts.update(
            classify_batch(
                batch=batch,
                query=settings["query"],
                model=settings["model"],
                api_key=settings["api_key"],
                base_url=settings["base_url"],
                timeout=settings["timeout"],
            )
        )

    kept: list[dict[str, Any]] = []
    for item in items:
        verdict = verdicts.get(item["id"], {})
        item["relevant"] = verdict.get("relevant", False)
        item["relevance"] = int(verdict.get("relevance", 0) or 0)
        item["importance"] = int(verdict.get("importance", 0) or 0)
        item["llm_summary"] = verdict.get("summary", "")
        item["reason"] = verdict.get("reason", "")
        if item["relevant"] and item["relevance"] >= settings["min_relevance"]:
            kept.append(item)

    kept.sort(
        key=lambda x: (
            int(x.get("importance", 0)),
            int(x.get("relevance", 0)),
            x.get("published") or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
        ),
        reverse=True,
    )
    kept = kept[: settings["top_k"]]

    return build_report_payload(
        query=settings["query"],
        kept=kept,
        total_items=len(items),
        days=settings["days"],
        model=settings["model"],
    )


def main() -> int:
    args = parse_args()
    settings = resolve_settings(args)

    if not settings["api_key"]:
        print(
            "Missing API key. Put it in config.toml under [llm].api_key, set NEWS_LLM_API_KEY / OPENAI_API_KEY, or pass --api-key.",
            file=sys.stderr,
        )
        return 2

    json_path, markdown_path, html_path = derive_paths(
        query=settings["query"],
        json_output=settings.get("output"),
        markdown_output=settings.get("markdown_output"),
        html_output=args.html_output,
    )

    report = investigate(settings)
    normalized_report = normalize_report(report)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved JSON report to {json_path}")

    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Saved markdown export to {markdown_path}")

    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(normalized_report), encoding="utf-8")
    print(f"Saved HTML visualization to {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
