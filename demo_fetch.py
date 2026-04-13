#!/usr/bin/env python3
"""Fetch RSS items and use an LLM to semantically filter and summarize them.

By default, the script looks for a config file next to itself at `config.toml`.
CLI flags override config values.

Example:
    python3 demo_fetch.py \
        --query "Important biotech and oncology developments this week" \
        --output reports/$(date +%F)-biotech-ai.json \
        --markdown-output reports/$(date +%F)-biotech-ai.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import textwrap
import tomllib
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_FEEDS = [
    ("FierceBiotech", "https://www.fiercebiotech.com/rss.xml"),
    ("Endpoints News", "https://endpoints.news/feed/"),
    ("STAT News", "https://www.statnews.com/feed/"),
    ("BioPharma Dive", "https://www.biopharmadive.com/feeds/news/"),
]

ATOM_NS = "{http://www.w3.org/2005/Atom}"
CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"
TZ = dt.timezone(dt.timedelta(hours=8))  # Asia/Shanghai for reporting
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")
SYSTEM_PROMPT = """You are helping build a personalized news digest.

You will receive:
1. a user request describing what news they want
2. a batch of article candidates with title, source, publish time, link, and a short RSS summary

Your job:
- judge semantic relevance to the request, not keyword overlap
- keep only articles a human would actually want in the digest
- prefer strong signal over weak maybe-matches
- never invent facts that are not in the title/summary

Return STRICT JSON with this schema:
{
  "results": [
    {
      "id": "article id",
      "relevant": true,
      "relevance": 0,
      "importance": 0,
      "summary": "<= 35 words, factual",
      "reason": "<= 20 words, why it matters to the request"
    }
  ]
}

Rules:
- relevance is 0-100 semantic fit to the user's request
- importance is 0-100 importance/usefulness IF included
- if evidence is weak, lower relevance
- include every provided id exactly once
- if not relevant, set relevant=false and still provide short summary/reason if possible
- output JSON only, no markdown fences or commentary"""


def fetch(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "NewsDeliveryAI/0.3 (+https://openclaw.local)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rsplit(" ", 1)[0].strip()
    return (clipped or text[: max_chars - 1]).rstrip() + "…"


def parse_date(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    except (TypeError, ValueError):
        pass
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    except ValueError:
        return None


def parse_rss_items(root: ET.Element) -> list[dict[str, Any]]:
    channel = root.find("channel")
    if channel is None:
        return []
    items = []
    for item in channel.findall("item"):
        title = strip_html(item.findtext("title"))
        link = strip_html(item.findtext("link"))
        summary = strip_html(
            item.findtext(f"{CONTENT_NS}encoded") or item.findtext("description")
        )
        pub = parse_date(item.findtext("pubDate") or item.findtext(f"{DC_NS}date"))
        items.append({"title": title, "link": link, "summary": summary, "published": pub})
    return items


def parse_atom_items(root: ET.Element) -> list[dict[str, Any]]:
    items = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = strip_html(entry.findtext(f"{ATOM_NS}title"))
        link = None
        for link_elem in entry.findall(f"{ATOM_NS}link"):
            rel = link_elem.attrib.get("rel", "alternate")
            if rel == "alternate" and link_elem.attrib.get("href"):
                link = link_elem.attrib["href"]
                break
        summary = strip_html(
            entry.findtext(f"{ATOM_NS}summary") or entry.findtext(f"{ATOM_NS}content")
        )
        pub_text = entry.findtext(f"{ATOM_NS}updated") or entry.findtext(f"{ATOM_NS}published")
        pub = parse_date(pub_text)
        items.append({"title": title, "link": link, "summary": summary, "published": pub})
    return items


def load_feed(source: str, url: str) -> list[dict[str, Any]]:
    raw = fetch(url)
    root = ET.fromstring(raw)
    if root.tag.lower().endswith("rss"):
        items = parse_rss_items(root)
    elif root.tag.startswith(ATOM_NS):
        items = parse_atom_items(root)
    else:
        items = parse_rss_items(root)
    for item in items:
        item["source"] = source
    return items


def dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for item in items:
        key = (item.get("link") or "").strip() or f"{item.get('source')}::{item.get('title')}"
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def batched(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def extract_message_text(data: dict[str, Any]) -> str:
    choice = data["choices"][0]["message"]["content"]
    if isinstance(choice, str):
        return choice
    if isinstance(choice, list):
        chunks = []
        for part in choice:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(part.get("text", ""))
        return "".join(chunks)
    raise ValueError("Unexpected response content shape")


def parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("rb") as fh:
        return tomllib.load(fh)


def config_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def get_feed_sources(config: dict[str, Any]) -> list[tuple[str, str]]:
    configured = config_get(config, "feeds", "sources", default=None)
    if not isinstance(configured, list):
        return DEFAULT_FEEDS

    feeds: list[tuple[str, str]] = []
    for row in configured:
        if not isinstance(row, dict):
            continue
        if row.get("enabled", True) is False:
            continue
        name = str(row.get("name", "")).strip()
        url = str(row.get("url", "")).strip()
        if name and url:
            feeds.append((name, url))
    return feeds or DEFAULT_FEEDS


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    settings = {
        "config_path": str(args.config),
        "query": args.query,
        "output": coalesce(
            args.output,
            config_get(config, "report", "default_json_output"),
            config_get(config, "digest", "default_output"),
        ),
        "markdown_output": coalesce(
            args.markdown_output,
            config_get(config, "report", "default_markdown_output"),
        ),
        "days": int(coalesce(args.days, config_get(config, "fetch", "days"), 7)),
        "batch_size": int(
            coalesce(args.batch_size, config_get(config, "fetch", "batch_size"), 10)
        ),
        "max_items": int(
            coalesce(args.max_items, config_get(config, "fetch", "max_items"), 50)
        ),
        "top_k": int(coalesce(args.top_k, config_get(config, "digest", "top_k"), 12)),
        "min_relevance": int(
            coalesce(args.min_relevance, config_get(config, "digest", "min_relevance"), 70)
        ),
        "model": str(
            coalesce(
                args.model,
                os.getenv("NEWS_LLM_MODEL"),
                os.getenv("OPENAI_MODEL"),
                config_get(config, "llm", "model"),
                "gpt-4.1-mini",
            )
        ),
        "base_url": str(
            coalesce(
                args.base_url,
                os.getenv("NEWS_LLM_BASE_URL"),
                os.getenv("OPENAI_BASE_URL"),
                config_get(config, "llm", "base_url"),
                "https://api.openai.com/v1",
            )
        ),
        "api_key": str(
            coalesce(
                args.api_key,
                os.getenv("NEWS_LLM_API_KEY"),
                os.getenv("OPENAI_API_KEY"),
                config_get(config, "llm", "api_key"),
                "",
            )
        ),
        "timeout": int(coalesce(args.timeout, config_get(config, "llm", "timeout"), 90)),
        "feeds": get_feed_sources(config),
    }
    return settings


def classify_batch(
    *,
    batch: list[dict[str, Any]],
    query: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout: int,
) -> dict[str, dict[str, Any]]:
    payload = {
        "request": query,
        "articles": [
            {
                "id": item["id"],
                "source": item.get("source"),
                "published": item.get("published_iso"),
                "title": item.get("title"),
                "summary": clip_text(item.get("summary") or "", 700),
                "link": item.get("link"),
            }
            for item in batch
        ],
    }

    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    parsed = parse_json_text(extract_message_text(response.json()))
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("Model response missing results list")

    mapped: dict[str, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            continue
        mapped[item_id] = {
            "relevant": bool(row.get("relevant", False)),
            "relevance": int(row.get("relevance", 0) or 0),
            "importance": int(row.get("importance", 0) or 0),
            "summary": str(row.get("summary", "")).strip(),
            "reason": str(row.get("reason", "")).strip(),
        }

    for item in batch:
        mapped.setdefault(
            item["id"],
            {
                "relevant": False,
                "relevance": 0,
                "importance": 0,
                "summary": "",
                "reason": "Model did not return a verdict for this item.",
            },
        )
    return mapped


def gather_items(days: int, feed_sources: list[tuple[str, str]]) -> list[dict[str, Any]]:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    collected: list[dict[str, Any]] = []
    for source, url in feed_sources:
        try:
            collected.extend(load_feed(source, url))
        except Exception as exc:
            print(f"Failed to load {source}: {exc}", file=sys.stderr)
    collected = dedupe_items(collected)
    filtered = []
    for item in collected:
        published = item.get("published")
        if not published or published < cutoff:
            continue
        filtered.append(item)
    filtered.sort(key=lambda x: x.get("published"), reverse=True)
    return filtered


def annotate_items(items: list[dict[str, Any]]) -> None:
    for idx, item in enumerate(items, start=1):
        item["id"] = f"a{idx:03d}"
        published = item.get("published")
        item["published_iso"] = published.astimezone(dt.timezone.utc).isoformat() if published else None


def build_report_payload(
    *,
    query: str,
    kept: list[dict[str, Any]],
    total_items: int,
    days: int,
    model: str,
) -> dict[str, Any]:
    generated_at = dt.datetime.now(TZ)
    articles = []
    for entry in kept:
        published = entry.get("published")
        articles.append(
            {
                "id": entry.get("id"),
                "title": entry.get("title"),
                "url": entry.get("link"),
                "source": entry.get("source"),
                "publishedAt": published.astimezone(TZ).isoformat() if published else None,
                "publishedLabel": published.astimezone(TZ).strftime("%Y-%m-%d %H:%M") if published else None,
                "relevance": int(entry.get("relevance", 0) or 0),
                "importance": int(entry.get("importance", 0) or 0),
                "summary": entry.get("llm_summary") or clip_text(entry.get("summary") or "", 180),
                "whyItMatters": entry.get("reason") or "Matched the requested topic.",
                "rssSummary": entry.get("summary") or "",
            }
        )

    return {
        "reportVersion": 1,
        "title": "AI-Filtered News Digest",
        "request": query,
        "generatedAt": generated_at.isoformat(),
        "timezone": "Asia/Shanghai",
        "coverage": {
            "days": days,
            "label": f"last {days} days",
        },
        "model": model,
        "stats": {
            "reviewed": total_items,
            "kept": len(articles),
        },
        "sources": sorted({article["source"] for article in articles if article.get("source")}),
        "articles": articles,
    }


def render_markdown(report: dict[str, Any]) -> str:
    generated = report.get("generatedAt") or ""
    lines = [
        f"# {report.get('title', 'AI-Filtered News Digest')}\n",
        f"Request: **{report.get('request', '')}**\n",
        f"Coverage window: {report.get('coverage', {}).get('label', 'unknown')} (generated {generated}, model `{report.get('model', '')}`).\n",
        f"Reviewed {report.get('stats', {}).get('reviewed', 0)} candidate articles, kept {report.get('stats', {}).get('kept', 0)}.\n",
    ]

    articles = report.get("articles") or []
    if not articles:
        lines.append("\nNo strong matches found.\n")
        return "\n".join(lines)

    for entry in articles:
        bullet = textwrap.dedent(
            f"""
            - **{entry.get('title')}** ({entry.get('source')} — {entry.get('publishedLabel') or 'Unknown'}, relevance {entry.get('relevance')}, importance {entry.get('importance')})  \
              {entry.get('summary')}  \
              Why it matters: {entry.get('whyItMatters') or 'Matched the requested topic.'}  \
              Link: {entry.get('url')}
            """
        ).strip()
        lines.append(bullet + "\n")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to TOML config file")
    parser.add_argument("--query", required=True, help="What kind of news the user wants")
    parser.add_argument("--output", help="Write the JSON report to this path")
    parser.add_argument("--markdown-output", help="Optional markdown export path")
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


def main() -> int:
    args = parse_args()
    settings = resolve_settings(args)

    if not settings["api_key"]:
        print(
            "Missing API key. Put it in config.toml under [llm].api_key, set NEWS_LLM_API_KEY / OPENAI_API_KEY, or pass --api-key.",
            file=sys.stderr,
        )
        return 2

    items = gather_items(settings["days"], settings["feeds"])
    if not items:
        print(json.dumps({"title": "AI-Filtered News Digest", "articles": []}, ensure_ascii=False, indent=2))
        return 0

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

    report = build_report_payload(
        query=settings["query"],
        kept=kept,
        total_items=len(items),
        days=settings["days"],
        model=settings["model"],
    )
    json_output = json.dumps(report, ensure_ascii=False, indent=2)

    if settings["output"]:
        output_path = Path(settings["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_output + "\n", encoding="utf-8")
        print(f"Saved JSON report to {output_path}")
    else:
        print(json_output)

    if settings["markdown_output"]:
        markdown_path = Path(settings["markdown_output"])
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Saved markdown export to {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
