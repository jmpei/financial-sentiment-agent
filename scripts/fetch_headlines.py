"""
Fetch live NewsAPI results for the domain-shift eval set.

Uses the agent's own search_news tool, so the texts are exactly what the model
sees in production (title + ". " + description). NewsAPI results change daily,
so the output is a frozen snapshot: labels are added to it by hand afterwards,
and domain_shift.py reads that labelled file. Re-running overwrites it.

Run:    .venv/bin/python -m scripts.fetch_headlines
Writes: domain_shift_headlines.jsonl  (label = null until labelled)
"""

import json

from src.tools import search_news

OUT_PATH = "domain_shift_headlines.jsonl"

QUERIES = [
    "Apple earnings", "Tesla stock", "NVIDIA shares", "Boeing", "JPMorgan",
    "Goldman Sachs", "Amazon stock", "Microsoft earnings", "Intel", "Pfizer",
    "Exxon Mobil", "Walmart", "Federal Reserve rates", "oil prices", "bank stocks",
    "layoffs", "IPO", "merger acquisition", "Royal Bank of Canada", "Shopify stock",
]


def main():
    seen, rows = set(), []
    for query in QUERIES:
        for a in search_news.func(query):
            title = a["title"].strip()
            if not title or title == "[Removed]" or title in seen:
                continue
            seen.add(title)
            text = f"{title}. {a['description']}" if a["description"] else title
            rows.append({
                "query":        query,
                "title":        title,
                "text":         text,
                "url":          a["url"],
                "publishedAt":  a["publishedAt"],
                "label":        None,
                "label_source": None,
            })
    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} unique articles from {len(QUERIES)} queries → {OUT_PATH}")


if __name__ == "__main__":
    main()
