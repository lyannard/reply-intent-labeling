#!/usr/bin/env python3
"""Label outbound-email replies with one of five buying-intent labels using a
free OpenRouter model.

Reads  : Q3_2026_Outbound_Email_Reply_Export.csv  (reply_id, account_id, reply_body, ...)
Writes : labels.csv  (reply_id, account_id, label, confidence, rationale), one row per reply

The API key is read from the OPENROUTER_API_KEY environment variable, which is
loaded from a .env file next to this script. The key is never written to any
other file, printed, or logged.

Usage:
    python label_replies.py                 # label everything
    python label_replies.py --dry-run       # show how many model calls it would make
    python label_replies.py --model <id>    # pin a specific :free model
    python label_replies.py --no-dedupe     # one call per reply (needs a higher quota)

Standard library only, no pip install needed.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

LABELS = ("ready_to_buy", "evaluating", "not_now", "not_interested", "wrong_contact")
OUT_COLUMNS = ["reply_id", "account_id", "label", "confidence", "rationale"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"
# openrouter/free routes each request to some free model. For repeatable labels,
# pin one specific ":free" model id from openrouter.ai/collections/free-models
# with --model or the OPENROUTER_MODEL environment variable.
DEFAULT_MODEL = "openrouter/free"
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_FILE = SCRIPT_DIR / ".label_cache.json"

SYSTEM_PROMPT = """You label replies to outbound sales emails for Notion (a docs and wiki workspace tool).
Give each reply exactly one buying-intent label from this list:

ready_to_buy: the sender asks for something that moves a purchase forward: a quote, pricing for a stated number of seats, an order form, a contract, a volume discount, or the process to upgrade their team.
evaluating: the sender is actively researching. They ask product, security or migration questions, or compare Notion with another tool, but make no purchase request yet.
not_now: no purchase action at the moment, but not a refusal. This includes timing or budget deferrals ("check back next quarter", "mid-reorg"), "the free plan is enough for now", and automatic replies (out of office, on leave, unmonitored mailbox), which say nothing about intent.
not_interested: a refusal. This includes a plain no, any request to unsubscribe, be removed or stop emailing (write "opt-out" in the rationale), and a sender who has already chosen another tool (Microsoft Loop, Confluence and similar).
wrong_contact: the sender is not the person who decides or buys. This includes an intern, someone who only uses it for personal notes, "ask our IT lead / procurement / office manager", forwarding to someone else, and individual students asking about personal or student pricing for themselves, a club or a study group (write "student" in the rationale).

Rules:
- Judge by what the sender asks for, not by tone.
- If the sender says they cannot decide, or is an individual student, use wrong_contact.
- A request to stop contact always wins: use not_interested.
- A request for a quote, order form, contract or upgrade process is ready_to_buy even if the sender also compares vendors.
- Automatic replies are not_now, and the rationale must start with "Auto-reply".

Respond with only a JSON object and nothing else:
{"label": "<one of the five labels>", "confidence": <number from 0 to 1>, "rationale": "<one sentence, under 25 words, citing what the reply says>"}"""


def load_env(path):
    """Minimal .env loader. Existing environment variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def normalize_key(body):
    """Cache key: collapse whitespace and replace digits, so replies that differ
    only in a number (seat counts, dates) are labeled with one model call."""
    return re.sub(r"\d+", "#", " ".join(body.split()))


def parse_label_json(text):
    """Turn the model's text into a validated {label, confidence, rationale}."""
    if not text:
        raise ValueError("empty response")
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in response")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    label = str(obj.get("label", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if label not in LABELS:
        raise ValueError("label is not one of the five: %r" % label)
    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        raise ValueError("confidence missing or not a number")
    confidence = round(min(1.0, max(0.0, confidence)), 2)
    rationale = " ".join(str(obj.get("rationale", "")).split())[:240]
    if not rationale:
        raise ValueError("rationale missing")
    return {"label": label, "confidence": confidence, "rationale": rationale}


def call_model(body, model, api_key, timeout=90):
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": 'Reply to label:\n"""\n%s\n"""' % body},
        ],
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    if "error" in data:
        raise RuntimeError("API error: %s" % data["error"])
    return data["choices"][0]["message"].get("content") or ""


def label_with_retries(body, model, api_key, retries=6):
    for attempt in range(1, retries + 1):
        try:
            return parse_label_json(call_model(body, model, api_key))
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                sys.exit("OpenRouter rejected the key (HTTP %d). Check OPENROUTER_API_KEY in .env." % err.code)
            if err.code in (400, 404):
                sys.exit("OpenRouter says the model or request is invalid (HTTP %d). "
                         "Check --model; free model ids change often." % err.code)
            problem = "HTTP %d" % err.code
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, RuntimeError) as err:
            problem = type(err).__name__
        if attempt == retries:
            sys.exit("Giving up after %d attempts (last problem: %s). If this was HTTP 429, the free "
                     "daily limit may be used up. Re-run later; finished calls are cached." % (retries, problem))
        wait = min(60, 5 * 2 ** (attempt - 1))
        print("  attempt %d failed (%s); retrying in %ds" % (attempt, problem, wait))
        time.sleep(wait)


def load_cache():
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def read_replies(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    missing = {"reply_id", "account_id", "reply_body"} - set(rows[0].keys() if rows else [])
    if missing:
        sys.exit("Input is missing required columns: %s" % ", ".join(sorted(missing)))
    ids = [r["reply_id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("reply_id values are not unique.")
    return rows


def compare_to_reference(out_path, ref_path):
    ref = {r["reply_id"]: r["label"] for r in csv.DictReader(open(ref_path, newline="", encoding="utf-8"))}
    new = list(csv.DictReader(open(out_path, newline="", encoding="utf-8")))
    diffs = Counter((ref[r["reply_id"]], r["label"]) for r in new if ref.get(r["reply_id"]) != r["label"])
    agree = len(new) - sum(diffs.values())
    print("\nAgreement with %s: %d of %d (%.1f%%)" % (ref_path.name, agree, len(new), 100.0 * agree / len(new)))
    for (was, now), n in diffs.most_common(15):
        print("  %3d replies: reference=%s, model=%s" % (n, was, now))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="Q3_2026_Outbound_Email_Reply_Export.csv")
    ap.add_argument("--out", default="labels.csv")
    ap.add_argument("--model", default=None, help="OpenRouter model id (default: $OPENROUTER_MODEL or %s)" % DEFAULT_MODEL)
    ap.add_argument("--delay", type=float, default=3.5, help="seconds between calls (free tier is 20 requests/minute)")
    ap.add_argument("--no-dedupe", action="store_true", help="one model call per reply instead of per distinct text")
    ap.add_argument("--dry-run", action="store_true", help="count the model calls needed and stop")
    args = ap.parse_args()

    load_env(SCRIPT_DIR / ".env")
    load_env(Path.cwd() / ".env")
    model = args.model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL

    input_path = Path(args.input)
    if not input_path.is_file() and (SCRIPT_DIR / args.input).is_file():
        input_path = SCRIPT_DIR / args.input
    if not input_path.is_file():
        sys.exit("Cannot find %s. Put the export next to this script or pass --input." % args.input)
    rows = read_replies(input_path)

    key_of = (lambda r: r["reply_id"]) if args.no_dedupe else (lambda r: normalize_key(r["reply_body"]))
    first_body = {}
    for r in rows:
        first_body.setdefault(key_of(r), r["reply_body"])
    print("%d replies, %d distinct texts to label, model: %s" % (len(rows), len(first_body), model))
    if args.dry_run:
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not found. Add a line OPENROUTER_API_KEY=... to the .env file in this folder.")

    cache = load_cache()
    done = cache.setdefault(model, {})
    todo = [k for k in first_body if k not in done]
    print("%d already cached, %d to call" % (len(first_body) - len(todo), len(todo)))
    for i, k in enumerate(todo, 1):
        done[k] = label_with_retries(first_body[k], model, api_key)
        save_cache(cache)
        print("  [%d/%d] %s" % (i, len(todo), done[k]["label"]))
        if i < len(todo):
            time.sleep(args.delay)

    out_path = Path(args.out)
    ref_path = out_path.with_name("labels_reference.csv")
    if out_path.exists() and not ref_path.exists():
        out_path.rename(ref_path)  # keep the checked reference labels for comparison
        print("Existing %s kept as %s" % (out_path.name, ref_path.name))

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for r in rows:
            res = done[key_of(r)]
            writer.writerow({"reply_id": r["reply_id"], "account_id": r["account_id"], **res})

    counts = Counter(done[key_of(r)]["label"] for r in rows)
    print("\nWrote %d rows to %s" % (len(rows), out_path))
    for label in LABELS:
        print("  %-15s %d" % (label, counts.get(label, 0)))
    if ref_path.exists():
        compare_to_reference(out_path, ref_path)


if __name__ == "__main__":
    main()
