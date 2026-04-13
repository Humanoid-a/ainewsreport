#!/usr/bin/env python3
"""Send a news report email via Resend.

Examples:
    python3 send_report_email.py reports/latest.json --to someone@example.com
    python3 send_report_email.py reports/latest.json --to a@example.com,b@example.com
"""

from __future__ import annotations

import argparse
import html
import sys
import tomllib
from pathlib import Path
from typing import Any

import requests

from visualize_report import load_report

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="JSON or markdown report to send")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to TOML config file")
    parser.add_argument("--to", action="append", help="Recipient email, repeatable or comma-separated")
    parser.add_argument("--subject", help="Email subject")
    parser.add_argument("--from-address", help="Override sender address")
    parser.add_argument("--from-name", help="Override sender display name")
    return parser.parse_args()


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


def parse_recipients(values: list[str] | None) -> list[str]:
    recipients: list[str] = []
    for value in values or []:
        for item in value.split(","):
            email = item.strip()
            if email:
                recipients.append(email)
    return recipients


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    recipients = parse_recipients(args.to)
    configured_recipients = config_get(config, "email", "default_recipients", default=[])
    if not recipients and isinstance(configured_recipients, list):
        recipients = [str(item).strip() for item in configured_recipients if str(item).strip()]

    return {
        "recipients": recipients,
        "from_address": str(
            coalesce(args.from_address, config_get(config, "email", "from_address"), "") or ""
        ),
        "from_name": str(
            coalesce(args.from_name, config_get(config, "email", "from_name"), "NewsDelivery")
        ),
        "subject_prefix": str(config_get(config, "email", "subject_prefix", default="[NewsDelivery]") or ""),
        "api_key": str(config_get(config, "email", "resend", "api_key", default="") or ""),
    }


def render_subject(report: dict[str, Any], subject_prefix: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    pieces = []
    if subject_prefix:
        pieces.append(subject_prefix)
    pieces.append(report.get("title") or "News Digest")
    request = report.get("request")
    if request:
        pieces.append(f"- {request}")
    return " ".join(pieces)


def sort_articles_by_time(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        articles,
        key=lambda article: (
            int(article.get("publishedEpochMs") or -1),
            int(article.get("importance") if article.get("importance") is not None else -1),
            int(article.get("relevance") if article.get("relevance") is not None else -1),
        ),
        reverse=True,
    )


def group_articles_by_time(report: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    articles = report.get("articles") or []
    reference_ms = report.get("generatedAtEpochMs")
    if not isinstance(reference_ms, (int, float)):
        timestamps = [article.get("publishedEpochMs") for article in articles if isinstance(article.get("publishedEpochMs"), (int, float))]
        reference_ms = max(timestamps) if timestamps else None

    if reference_ms is None:
        grouped = sort_articles_by_time(list(articles))
        return [("Latest news, last 24 hours", grouped)] if grouped else []

    day_ms = 24 * 60 * 60 * 1000
    week_ms = 7 * day_ms
    latest: list[dict[str, Any]] = []
    this_week: list[dict[str, Any]] = []
    older: list[dict[str, Any]] = []

    for article in articles:
        published_ms = article.get("publishedEpochMs")
        if not isinstance(published_ms, (int, float)):
            older.append(article)
            continue
        age_ms = reference_ms - published_ms
        if age_ms <= day_ms:
            latest.append(article)
        elif age_ms <= week_ms:
            this_week.append(article)
        else:
            older.append(article)

    sections = [
        ("Latest news, last 24 hours", sort_articles_by_time(latest)),
        ("Earlier this week", sort_articles_by_time(this_week)),
        ("Older / undated", sort_articles_by_time(older)),
    ]
    return [(title, items) for title, items in sections if items]


def render_text_email(report: dict[str, Any]) -> str:
    lines = [report.get("title") or "News Digest", ""]
    if report.get("request"):
        lines.append(f"Request: {report['request']}")
    if report.get("coverage"):
        lines.append(f"Coverage: {report['coverage']}")
    if report.get("reviewed"):
        lines.append(report["reviewed"])
    lines.append("")

    for section_title, articles in group_articles_by_time(report):
        lines.append("")
        lines.append(section_title)
        lines.append("-" * len(section_title))
        lines.append("")
        for article in articles:
            lines.append(article.get("title") or "(untitled)")
            meta = []
            if article.get("source"):
                meta.append(article["source"])
            if article.get("published"):
                meta.append(article["published"])
            if article.get("relevance") is not None:
                meta.append(f"relevance {article['relevance']}")
            if article.get("importance") is not None:
                meta.append(f"importance {article['importance']}")
            if meta:
                lines.append(" | ".join(meta))
            if article.get("summary"):
                lines.append(article["summary"])
            if article.get("reason"):
                lines.append(f"Why it matters: {article['reason']}")
            if article.get("link"):
                lines.append(article["link"])
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_html_email(report: dict[str, Any]) -> str:
    title = html.escape(report.get("title") or "News Digest")
    request = html.escape(report.get("request") or "")
    coverage = html.escape(report.get("coverage") or "")
    reviewed = html.escape(report.get("reviewed") or "")

    section_rows = []
    for section_title, articles in group_articles_by_time(report):
        cards = []
        for article in articles:
            title_html = html.escape(article.get("title") or "Untitled")
            link = article.get("link")
            if link:
                title_html = f'<a href="{html.escape(link)}" style="color:#0f62fe;text-decoration:none;">{title_html}</a>'

            meta_bits = []
            if article.get("source"):
                meta_bits.append(html.escape(article["source"]))
            if article.get("published"):
                meta_bits.append(html.escape(article["published"]))
            if article.get("relevance") is not None:
                meta_bits.append(f"Relevance {html.escape(str(article['relevance']))}")
            if article.get("importance") is not None:
                meta_bits.append(f"Importance {html.escape(str(article['importance']))}")
            meta_html = " · ".join(meta_bits)

            summary_html = html.escape(article.get("summary") or "")
            reason_html = html.escape(article.get("reason") or "")

            cards.append(
                f"""
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #d8e0f0;border-radius:12px;margin:0 0 14px 0;">
                  <tr>
                    <td style="padding:16px 18px;font-family:Arial,sans-serif;">
                      <div style="font-size:18px;font-weight:700;line-height:1.3;color:#111827;">{title_html}</div>
                      <div style="margin-top:6px;font-size:12px;line-height:1.4;color:#5b6475;">{meta_html}</div>
                      <div style="margin-top:12px;font-size:14px;line-height:1.6;color:#1f2937;">{summary_html}</div>
                      {f'<div style="margin-top:10px;font-size:13px;line-height:1.5;color:#5b6475;">Why it matters: {reason_html}</div>' if reason_html else ''}
                    </td>
                  </tr>
                </table>
                """.strip()
            )

        section_rows.append(
            f"""
            <tr>
              <td style="padding:0 0 24px 0;">
                <div style="padding:0 0 12px 0;font-family:Arial,sans-serif;font-size:20px;font-weight:800;line-height:1.3;color:#111827;">{html.escape(section_title)}</div>
                {''.join(cards)}
              </td>
            </tr>
            """.strip()
        )

    return f"""<!doctype html>
<html>
  <body style="margin:0;padding:24px;background:#f3f6fb;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:760px;margin:0 auto;background:#ffffff;border-radius:16px;border:1px solid #d8e0f0;overflow:hidden;">
      <tr>
        <td style="padding:28px 28px 18px 28px;background:linear-gradient(135deg,#eaf2ff,#eefcf6);font-family:Arial,sans-serif;">
          <div style="font-size:28px;font-weight:800;line-height:1.2;color:#111827;">{title}</div>
          {f'<div style="margin-top:8px;font-size:15px;line-height:1.5;color:#334155;">Request: {request}</div>' if request else ''}
          {f'<div style="margin-top:10px;font-size:13px;line-height:1.5;color:#5b6475;">{coverage}</div>' if coverage else ''}
          {f'<div style="margin-top:4px;font-size:13px;line-height:1.5;color:#5b6475;">{reviewed}</div>' if reviewed else ''}
        </td>
      </tr>
      <tr>
        <td style="padding:24px 28px 10px 28px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
            {''.join(section_rows)}
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


def send_email(*, settings: dict[str, Any], report: dict[str, Any], explicit_subject: str | None) -> str:
    if not settings["api_key"]:
        raise ValueError("Missing Resend API key in config.toml [email.resend].")
    if not settings["from_address"]:
        raise ValueError(
            "Missing email.from_address in config.toml. Resend requires a verified sender address or domain."
        )
    if not settings["recipients"]:
        raise ValueError("No recipients provided. Use --to or set email.default_recipients in config.toml.")

    subject = render_subject(report, settings["subject_prefix"], explicit_subject)
    response = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "from": f'{settings["from_name"]} <{settings["from_address"]}>',
            "to": settings["recipients"],
            "subject": subject,
            "html": render_html_email(report),
            "text": render_text_email(report),
        },
        timeout=30,
    )
    if not response.ok:
        detail = ""
        try:
            data = response.json()
            detail = data.get("message") or data.get("error") or response.text
        except Exception:
            detail = response.text
        raise ValueError(f"Resend API request failed: {detail.strip()}")

    data = response.json()
    return str(data.get("id") or "")


def main() -> int:
    args = parse_args()
    settings = resolve_settings(args)
    report = load_report(Path(args.report))
    try:
        message_id = send_email(settings=settings, report=report, explicit_subject=args.subject)
    except Exception as exc:
        print(f"Failed to send email: {exc}", file=sys.stderr)
        return 1
    print(f"Sent report to {', '.join(settings['recipients'])} via resend (id: {message_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
