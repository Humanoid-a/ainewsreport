#!/usr/bin/env python3
"""Render a JSON or markdown news digest into a self-contained interactive HTML page.

Example:
    python3 visualize_report.py \
        reports/2026-04-12-biotech-ai.json \
        --output web/2026-04-12-biotech-ai.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="JSON or markdown report to visualize")
    parser.add_argument("--output", help="Where to write the HTML file")
    return parser.parse_args()


def clean_line(line: str) -> str:
    return line.strip().rstrip("\\").strip()


def coerce_tzinfo(value: Any):
    if value is None:
        return None
    if isinstance(value, timezone):
        return value
    if hasattr(value, "utcoffset"):
        return value

    text = str(value).strip()
    if not text:
        return None

    offset_match = re.fullmatch(r"(?:UTC)?([+-])(\d{2}):(\d{2})", text)
    if offset_match:
        sign = 1 if offset_match.group(1) == "+" else -1
        hours = int(offset_match.group(2))
        minutes = int(offset_match.group(3))
        return timezone(sign * timedelta(hours=hours, minutes=minutes))

    if ZoneInfo is not None:
        try:
            return ZoneInfo(text)
        except Exception:
            return None
    return None


def parse_datetimeish(value: Any, default_tz: Any = None) -> datetime | None:
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None

    if parsed is None:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        return None

    if parsed.tzinfo is None:
        tzinfo = coerce_tzinfo(default_tz)
        if tzinfo is not None:
            parsed = parsed.replace(tzinfo=tzinfo)
    return parsed


def datetime_to_epoch_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def extract_generated_at(coverage: Any) -> str | None:
    if not coverage:
        return None
    text = str(coverage)
    match = re.search(r"generated\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})(?::(\d{2}))?\s+UTC([+-]\d{2}:\d{2})", text)
    if not match:
        return None
    seconds = match.group(3) or "00"
    return f"{match.group(1)}T{match.group(2)}:{seconds}{match.group(4)}"


def normalize_article(article: dict[str, Any], default_tz: Any = None) -> dict[str, Any]:
    published_display = article.get("publishedLabel") or article.get("published") or article.get("publishedAt")
    published_raw = article.get("publishedAt") or article.get("publishedLabel") or article.get("published")
    published_dt = parse_datetimeish(published_raw, default_tz=default_tz)
    return {
        "title": article.get("title"),
        "source": article.get("source"),
        "published": published_display,
        "publishedAt": article.get("publishedAt") or published_raw,
        "publishedEpochMs": datetime_to_epoch_ms(published_dt),
        "relevance": article.get("relevance"),
        "importance": article.get("importance"),
        "summary": article.get("summary"),
        "reason": article.get("whyItMatters") or article.get("reason"),
        "link": article.get("url") or article.get("link"),
    }


def parse_meta(meta: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": None,
        "published": None,
        "relevance": None,
        "importance": None,
    }
    if not meta:
        return result

    pieces = [piece.strip() for piece in meta.split(",")]
    head = pieces[0] if pieces else ""
    if " — " in head:
        source, published = head.split(" — ", 1)
        result["source"] = source.strip() or None
        result["published"] = published.strip() or None
    else:
        result["source"] = head or None

    for piece in pieces[1:]:
        lowered = piece.lower()
        if lowered.startswith("relevance"):
            match = re.search(r"(\d+)", piece)
            if match:
                result["relevance"] = int(match.group(1))
        elif lowered.startswith("importance"):
            match = re.search(r"(\d+)", piece)
            if match:
                result["importance"] = int(match.group(1))
    return result


def parse_article_block(lines: list[str]) -> dict[str, Any]:
    first = clean_line(lines[0])
    match = re.match(r"^- \*\*(.+?)\*\* \((.+?)\)\s*(.*)$", first)
    if not match:
        raise ValueError(f"Unrecognized article header: {first}")

    title = match.group(1).strip()
    meta = parse_meta(match.group(2).strip())
    body_parts: list[str] = []
    if match.group(3).strip():
        body_parts.append(match.group(3).strip())

    for raw in lines[1:]:
        line = clean_line(raw)
        if line:
            body_parts.append(line)

    body = " ".join(body_parts).strip()

    link = None
    link_match = re.search(r"Link:\s*(https?://\S+)", body)
    if link_match:
        link = link_match.group(1).strip()
        body = (body[: link_match.start()] + " " + body[link_match.end() :]).strip()

    reason = None
    reason_match = re.search(r"Why it matters:\s*(.+)$", body)
    if reason_match:
        reason = reason_match.group(1).strip()
        body = body[: reason_match.start()].strip()

    summary = re.sub(r"\s+", " ", body).strip()

    return {
        "title": title,
        "source": meta["source"],
        "published": meta["published"],
        "relevance": meta["relevance"],
        "importance": meta["importance"],
        "summary": summary,
        "reason": reason,
        "link": link,
    }


def parse_markdown_report(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    title = None
    request = None
    coverage = None
    reviewed = None

    article_blocks: list[list[str]] = []
    current_block: list[str] = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("Request:") and not request:
            request = line.split(":", 1)[1].strip().strip("*")
            continue
        if line.startswith("Coverage window:") and not coverage:
            coverage = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Reviewed ") and not reviewed:
            reviewed = line.strip()
            continue
        if line.startswith("- **"):
            if current_block:
                article_blocks.append(current_block)
            current_block = [line]
            continue
        if current_block:
            current_block.append(line)

    if current_block:
        article_blocks.append(current_block)

    articles = [parse_article_block(block) for block in article_blocks]
    sources = sorted({article["source"] for article in articles if article.get("source")})

    return {
        "title": title or "News Digest",
        "request": request,
        "coverage": coverage,
        "reviewed": reviewed,
        "articleCount": len(articles),
        "sources": sources,
        "articles": articles,
    }


def normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report.get("articles"), list):
        raise ValueError("Unsupported report format")

    coverage = report.get("coverage") or {}
    if isinstance(coverage, dict):
        coverage_text = coverage.get("label") or coverage.get("days")
    else:
        coverage_text = coverage

    generated_at = report.get("generatedAt") or extract_generated_at(coverage_text)
    default_tz = report.get("timezone") or (parse_datetimeish(generated_at).tzinfo if generated_at else None)
    articles = [normalize_article(article, default_tz=default_tz) for article in report.get("articles") or [] if isinstance(article, dict)]

    stats = report.get("stats") or {}
    reviewed = report.get("reviewed")
    if not reviewed and isinstance(stats, dict) and stats:
        reviewed = f"Reviewed {stats.get('reviewed', 0)} candidate articles, kept {stats.get('kept', len(articles))}."

    sources = report.get("sources")
    if not isinstance(sources, list):
        sources = sorted({article["source"] for article in articles if article.get("source")})

    generated_dt = parse_datetimeish(generated_at, default_tz=default_tz)

    return {
        "title": report.get("title") or "News Digest",
        "request": report.get("request"),
        "coverage": coverage_text,
        "reviewed": reviewed,
        "generatedAt": generated_at,
        "generatedAtEpochMs": datetime_to_epoch_ms(generated_dt),
        "timezone": report.get("timezone"),
        "articleCount": len(articles),
        "sources": sources,
        "articles": articles,
    }


def load_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if path.suffix.lower() == ".json" or stripped.startswith("{"):
        return normalize_report(json.loads(text))
    return normalize_report(parse_markdown_report(text))


def render_html(report: dict[str, Any]) -> str:
    data_json = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(report.get("title") or "News Digest")
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0b1020;
      --panel: #121931;
      --panel-2: #18213f;
      --text: #ecf2ff;
      --muted: #a9b5d4;
      --accent: #70a5ff;
      --accent-2: #7ef0c2;
      --border: rgba(255,255,255,0.08);
      --chip: rgba(255,255,255,0.08);
      --shadow: 0 20px 50px rgba(0,0,0,0.25);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #0a0f1f 0%, #0e1730 100%);
      color: var(--text);
    }}
    .wrap {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 64px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(112,165,255,0.16), rgba(126,240,194,0.10));
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 44px); line-height: 1.05; }}
    .sub {{ color: var(--muted); margin: 8px 0 0; line-height: 1.5; }}
    .meta-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .chip {{
      background: var(--chip);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .stat {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px 18px;
    }}
    .stat-label {{ color: var(--muted); font-size: 13px; }}
    .stat-value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
    .controls {{
      margin-top: 22px;
      display: grid;
      grid-template-columns: 2fr 1fr 1fr;
      gap: 12px;
    }}
    .control {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }}
    .control label {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .control input, .control select {{
      width: 100%;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 15px;
    }}
    .range-line {{ display: flex; align-items: center; gap: 10px; }}
    .range-line input {{ flex: 1; }}
    .range-value {{ min-width: 36px; text-align: right; color: var(--accent-2); font-weight: 600; }}
    .list {{ display: grid; gap: 28px; margin-top: 22px; }}
    .section {{ display: grid; gap: 16px; }}
    .section-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      padding: 0 4px;
    }}
    .section-title {{ margin: 0; font-size: 22px; }}
    .section-sub {{ color: var(--muted); font-size: 14px; }}
    .card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 20px;
      box-shadow: var(--shadow);
    }}
    .card-head {{ display: flex; gap: 16px; justify-content: space-between; align-items: flex-start; }}
    .card h2 {{ margin: 0; font-size: 22px; line-height: 1.2; }}
    .card a {{ color: var(--text); text-decoration: none; }}
    .card a:hover {{ color: var(--accent); }}
    .card-meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .badge {{
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 13px;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      color: var(--muted);
    }}
    .score-badge {{ color: var(--text); background: rgba(112,165,255,0.12); }}
    .importance-badge {{ color: var(--text); background: rgba(126,240,194,0.12); }}
    .summary {{ margin: 16px 0 10px; color: #f5f8ff; line-height: 1.6; }}
    .reason {{ color: var(--muted); font-size: 14px; }}
    .empty {{
      margin-top: 22px;
      padding: 28px;
      border-radius: 20px;
      background: rgba(255,255,255,0.04);
      border: 1px dashed var(--border);
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 860px) {{
      .controls {{ grid-template-columns: 1fr; }}
      .card-head {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <h1 id=\"report-title\"></h1>
      <p class=\"sub\" id=\"report-request\"></p>
      <div class=\"meta-row\" id=\"meta-row\"></div>
      <div class=\"stats\">
        <div class=\"stat\"><div class=\"stat-label\">Articles</div><div class=\"stat-value\" id=\"stat-count\">0</div></div>
        <div class=\"stat\"><div class=\"stat-label\">Sources</div><div class=\"stat-value\" id=\"stat-sources\">0</div></div>
        <div class=\"stat\"><div class=\"stat-label\">Avg relevance</div><div class=\"stat-value\" id=\"stat-relevance\">—</div></div>
      </div>
    </section>

    <section class=\"controls\">
      <div class=\"control\">
        <label for=\"search\">Search</label>
        <input id=\"search\" type=\"search\" placeholder=\"Search titles, summaries, reasons...\">
      </div>
      <div class=\"control\">
        <label for=\"source\">Source</label>
        <select id=\"source\">
          <option value=\"\">All sources</option>
        </select>
      </div>
      <div class=\"control\">
        <label for=\"relevance\">Minimum relevance</label>
        <div class=\"range-line\">
          <input id=\"relevance\" type=\"range\" min=\"0\" max=\"100\" value=\"0\">
          <div class=\"range-value\" id=\"relevance-value\">0</div>
        </div>
      </div>
    </section>

    <section class=\"list\" id=\"list\"></section>
    <div class=\"empty\" id=\"empty\" hidden>No articles match the current filters.</div>
  </div>

  <script id=\"report-data\" type=\"application/json\">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('report-data').textContent);
    const els = {{
      title: document.getElementById('report-title'),
      request: document.getElementById('report-request'),
      metaRow: document.getElementById('meta-row'),
      statCount: document.getElementById('stat-count'),
      statSources: document.getElementById('stat-sources'),
      statRelevance: document.getElementById('stat-relevance'),
      source: document.getElementById('source'),
      search: document.getElementById('search'),
      relevance: document.getElementById('relevance'),
      relevanceValue: document.getElementById('relevance-value'),
      list: document.getElementById('list'),
      empty: document.getElementById('empty'),
    }};

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function chip(text) {{
      return `<span class=\"chip\">${{escapeHtml(text)}}</span>`;
    }}

    function badge(text, className='') {{
      return `<span class=\"badge ${{className}}\">${{escapeHtml(text)}}</span>`;
    }}

    function renderHeader() {{
      els.title.textContent = data.title || 'News Digest';
      els.request.textContent = data.request ? `Request: ${{data.request}}` : 'Interactive report view';
      const meta = [];
      if (data.coverage) meta.push(chip(data.coverage));
      if (data.reviewed) meta.push(chip(data.reviewed));
      els.metaRow.innerHTML = meta.join('');

      const withRelevance = data.articles.filter(a => Number.isFinite(a.relevance));
      const avgRelevance = withRelevance.length
        ? Math.round(withRelevance.reduce((sum, a) => sum + a.relevance, 0) / withRelevance.length)
        : null;
      els.statCount.textContent = String(data.articleCount || data.articles.length || 0);
      els.statSources.textContent = String((data.sources || []).length);
      els.statRelevance.textContent = avgRelevance === null ? '—' : String(avgRelevance);

      for (const source of data.sources || []) {{
        const option = document.createElement('option');
        option.value = source;
        option.textContent = source;
        els.source.appendChild(option);
      }}
    }}

    function matches(article) {{
      const term = els.search.value.trim().toLowerCase();
      const source = els.source.value;
      const minRelevance = Number(els.relevance.value || 0);
      const articleRelevance = Number.isFinite(article.relevance) ? article.relevance : 0;
      if (source && article.source !== source) return false;
      if (articleRelevance < minRelevance) return false;
      if (!term) return true;
      const haystack = [article.title, article.summary, article.reason, article.source]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      return haystack.includes(term);
    }}

    function articleCard(article) {{
      const meta = [];
      if (article.source) meta.push(badge(article.source));
      if (article.published) meta.push(badge(article.published));
      if (Number.isFinite(article.relevance)) meta.push(badge(`Relevance ${{article.relevance}}`, 'score-badge'));
      if (Number.isFinite(article.importance)) meta.push(badge(`Importance ${{article.importance}}`, 'importance-badge'));
      const title = article.link
        ? `<a href=\"${{escapeHtml(article.link)}}\" target=\"_blank\" rel=\"noreferrer\">${{escapeHtml(article.title)}}</a>`
        : escapeHtml(article.title);
      const summary = article.summary ? `<p class=\"summary\">${{escapeHtml(article.summary)}}</p>` : '';
      const reason = article.reason ? `<div class=\"reason\">Why it matters: ${{escapeHtml(article.reason)}}</div>` : '';
      return `
        <article class=\"card\">
          <div class=\"card-head\">
            <div>
              <h2>${{title}}</h2>
              <div class=\"card-meta\">${{meta.join('')}}</div>
            </div>
          </div>
          ${{summary}}
          ${{reason}}
        </article>
      `;
    }}

    function sectionBlock(title, subtitle, articles) {{
      if (!articles.length) return '';
      return `
        <section class="section">
          <div class="section-head">
            <h2 class="section-title">${{escapeHtml(title)}}</h2>
            <div class="section-sub">${{escapeHtml(subtitle)}}</div>
          </div>
          ${{articles.map(articleCard).join('')}}
        </section>
      `;
    }}

    function getReferenceTimeMs(articles) {{
      if (Number.isFinite(data.generatedAtEpochMs)) return data.generatedAtEpochMs;
      const timestamps = articles.map(article => article.publishedEpochMs).filter(Number.isFinite);
      if (timestamps.length) return Math.max(...timestamps);
      return Date.now();
    }}

    function compareByTimeDesc(a, b) {{
      const aTime = Number.isFinite(a.publishedEpochMs) ? a.publishedEpochMs : -1;
      const bTime = Number.isFinite(b.publishedEpochMs) ? b.publishedEpochMs : -1;
      return bTime - aTime || (b.importance ?? -1) - (a.importance ?? -1) || (b.relevance ?? -1) - (a.relevance ?? -1);
    }}

    function renderList() {{
      els.relevanceValue.textContent = els.relevance.value;
      const filtered = (data.articles || []).filter(matches);
      const referenceTimeMs = getReferenceTimeMs(filtered);
      const dayMs = 24 * 60 * 60 * 1000;
      const weekMs = 7 * dayMs;

      const latest = [];
      const thisWeek = [];
      const older = [];

      for (const article of filtered) {{
        if (!Number.isFinite(article.publishedEpochMs)) {{
          older.push(article);
          continue;
        }}
        const ageMs = referenceTimeMs - article.publishedEpochMs;
        if (ageMs <= dayMs) latest.push(article);
        else if (ageMs <= weekMs) thisWeek.push(article);
        else older.push(article);
      }}

      latest.sort(compareByTimeDesc);
      thisWeek.sort(compareByTimeDesc);
      older.sort(compareByTimeDesc);

      const sections = [
        sectionBlock('Latest news, last 24 hours', `${{latest.length}} article${{latest.length === 1 ? '' : 's'}}`, latest),
        sectionBlock('Earlier this week', `${{thisWeek.length}} article${{thisWeek.length === 1 ? '' : 's'}}`, thisWeek),
        sectionBlock('Older / undated', `${{older.length}} article${{older.length === 1 ? '' : 's'}}`, older),
      ].filter(Boolean);

      els.list.innerHTML = sections.join('');
      els.empty.hidden = filtered.length > 0;
    }}

    renderHeader();
    renderList();
    els.search.addEventListener('input', renderList);
    els.source.addEventListener('change', renderList);
    els.relevance.addEventListener('input', renderList);
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    report_path = Path(args.report)
    output_path = Path(args.output) if args.output else report_path.with_suffix(".html")

    report = load_report(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(report), encoding="utf-8")
    print(f"Saved HTML report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
