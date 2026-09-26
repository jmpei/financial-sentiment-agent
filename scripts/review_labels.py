"""
Review the domain-shift labels one row at a time, one keypress per row.

Enter keeps the current label, 1-5 replace it; either way the row becomes
label_source = "author" and its pre-review label is kept in `draft_label`, so
agreement with the draft can be reported. Every keypress is saved to disk, and
a restart resumes at the first row not yet reviewed.

Run:    .venv/bin/python -m scripts.review_labels [path]
Edits:  domain_shift_headlines.jsonl (in place)
"""

import json
import os
import shutil
import sys
import termios
import textwrap
import tty

PATH = sys.argv[1] if len(sys.argv) > 1 else "domain_shift_headlines.jsonl"

KEYS = {"1": "positive", "2": "neutral", "3": "negative", "4": "off_topic", "5": "duplicate"}
COLOR = {"positive": "32", "neutral": "34", "negative": "31", "off_topic": "90", "duplicate": "90"}
ENTER = ("\r", "\n")


def getch() -> str:
    """One keypress; main() keeps the terminal in cbreak mode for the whole session."""
    return sys.stdin.read(1)


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def save(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def review(row, label):
    row.setdefault("draft_label", row["label"])
    row["label"] = label
    row["label_source"] = "author"


def _paint(label) -> str:
    return f"\033[1;{COLOR[label]}m{label.upper()}\033[0m"


def show(rows, i):
    r = rows[i]
    width = min(shutil.get_terminal_size().columns, 100) - 4
    done = sum(x["label_source"] == "author" for x in rows)
    desc = r["text"][len(r["title"]):].lstrip(". ").strip()
    source = "已复核" if r["label_source"] == "author" else "草稿"

    out = ["\033[2J\033[H"]
    out.append(f"  {i + 1} / {len(rows)}      已复核 {done}      搜索词：{r['query']}\n\n")
    out += [f"  \033[1m{line}\033[0m\n" for line in textwrap.wrap(r["title"], width)]
    out.append("\n")
    out += [f"  {line}\n" for line in textwrap.wrap(" ".join(desc.split()), width)]
    out.append(f"\n  当前标签：{_paint(r['label'])}   （{source}）\n\n")
    out.append("  标准：站在投资者角度，这条新闻对它说的公司或市场是利好、利空，还是中性\n\n")
    out.append("  Enter 同意\n")
    out.append("  " + "   ".join(f"{k} {_paint(v)}" for k, v in KEYS.items()) + "\n")
    out.append("  b 上一条      q 退出（已自动保存）\n")
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def summary(rows):
    reviewed = [r for r in rows if r["label_source"] == "author"]
    changed = [r for r in reviewed if r["label"] != r["draft_label"]]
    print(f"\n  已复核 {len(reviewed)} / {len(rows)} 条，改了 {len(changed)} 条。")
    if reviewed:
        print(f"  和草稿一致：{1 - len(changed) / len(reviewed):.0%}")
    if len(reviewed) == len(rows):
        print("  全部复核完。下一步：.venv/bin/python domain_shift.py")
    else:
        print("  下次运行会从第一条没复核的接着来。")


def main():
    rows = load(PATH)
    i = next((k for k, r in enumerate(rows) if r["label_source"] != "author"), len(rows))
    # cbreak once for the session: toggling it per key would drop keys typed mid-redraw
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while i < len(rows):
            show(rows, i)
            key = getch()
            if key in ENTER or key in KEYS:
                review(rows[i], rows[i]["label"] if key in ENTER else KEYS[key])
                save(PATH, rows)
                i += 1
            elif key == "b":
                i = max(i - 1, 0)
            elif key in ("q", "\x03"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    sys.stdout.write("\033[2J\033[H")
    summary(rows)


if __name__ == "__main__":
    main()
