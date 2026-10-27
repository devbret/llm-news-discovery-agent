import json
import logging
import os
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, render_template, request, stream_with_context

load_dotenv()

DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", "data"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
CHAT_TIMEOUT = int(os.getenv("CHAT_TIMEOUT", "300"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
CHAT_CONTEXT_CHARS = int(os.getenv("CHAT_CONTEXT_CHARS", "20000"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))

RUN_FILE_RE = re.compile(r"^news_agent_log_(\d+)\.json$")
RUN_ID_RE = re.compile(r"^\d{1,20}$")
MAX_HISTORY_MESSAGES = 30
MAX_SUPER_HOT_IN_CONTEXT = 40
MAX_FOCUS_ARTICLES = 5
MAX_HOURLY_SPAN_DAYS = 21
TOP_SOURCES = 12

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dashboard")

app = Flask(__name__)
app.json.sort_keys = False

_cache_lock = threading.Lock()
_payload_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def run_path(run_id: str) -> str:
    return os.path.join(DATA_DIR, f"news_agent_log_{run_id}.json")


def summary_path(run_id: str) -> str:
    return os.path.join(DATA_DIR, f"news_summary_{run_id}.json")


def discover_runs() -> List[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    try:
        names = os.listdir(DATA_DIR)
    except OSError as e:
        logger.warning(f"Cannot list DATA_DIR {DATA_DIR}: {e}")
        return []
    ids = [m.group(1) for n in names if (m := RUN_FILE_RE.match(n))]
    return sorted(ids, key=int, reverse=True)


def load_summary(run_id: str) -> Optional[Dict[str, Any]]:
    try:
        with open(summary_path(run_id), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("text"):
        return None
    return data


def save_summary(run_id: str, model: str, text: str) -> None:
    payload = {"model": model, "created_at": datetime.now(timezone.utc).isoformat(), "text": text}
    with open(summary_path(run_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved run summary for {run_id}")


def parse_view_id(view_id: str) -> List[str]:
    ids: List[str] = []
    for part in str(view_id or "").split("+"):
        part = part.strip()
        if part and RUN_ID_RE.match(part) and part not in ids:
            ids.append(part)
    return sorted(ids, key=int)


def load_payload(view_id: str) -> Optional[Dict[str, Any]]:
    ids = parse_view_id(view_id)
    if not ids:
        return None
    view_key = "+".join(ids)
    stamps: List[Tuple[str, float]] = []
    for run_id in ids:
        try:
            stamps.append((run_id, os.path.getmtime(run_path(run_id))))
        except OSError:
            logger.warning(f"Run file missing for {run_id}")
    if not stamps:
        return None
    signature = tuple(stamps)

    with _cache_lock:
        cached = _payload_cache.get(view_key)
        if cached and cached[0] == signature:
            return cached[1]

    loaded: List[Tuple[str, Dict[str, Any]]] = []
    for run_id, _ in stamps:
        try:
            with open(run_path(run_id), encoding="utf-8") as f:
                log = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning(f"Skipping unreadable run file {run_path(run_id)}: {e}")
            continue
        if isinstance(log, dict):
            loaded.append((run_id, log))
    if not loaded:
        return None

    payload = build_payload(view_key, loaded)
    with _cache_lock:
        _payload_cache[view_key] = (signature, payload)
    return payload


def normalize_keyword(keyword: Any) -> str:
    return str(keyword or "").strip().lower()


def is_placeholder(article: Dict[str, Any]) -> bool:
    title = str(article.get("title") or "").strip()
    url = str(article.get("url") or "")
    return title == "[Removed]" or url.startswith("https://removed.com")


def parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fill_buckets(counter: Counter, hot: Counter, start: datetime, end: datetime,
                 step: timedelta, fmt: str) -> List[Dict[str, Any]]:
    out = []
    cur = start
    while cur <= end:
        key = cur.strftime(fmt)
        out.append({"bucket": key, "count": counter.get(key, 0), "super_hot": hot.get(key, 0)})
        cur += step
    return out


def build_tree(ks: List[Dict[str, Any]], labels: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], set]:
    searched_index: Dict[str, Dict[str, Any]] = {}
    for k in ks:
        searched_index.setdefault(normalize_keyword(k["keyword"]), k)
    parent_of: Dict[str, str] = {}
    children: Dict[str, List[str]] = defaultdict(list)
    suggested: set = set()

    for k in ks:
        parent_norm = normalize_keyword(k["keyword"])
        for rel in k["related_keywords"]:
            rel_norm = normalize_keyword(rel)
            if not rel_norm:
                continue
            suggested.add(rel_norm)
            labels.setdefault(rel_norm, rel)
            target = searched_index.get(rel_norm)
            if target is None:
                if rel_norm not in children[parent_norm]:
                    children[parent_norm].append(rel_norm)
            elif target["order"] > k["order"] and rel_norm not in parent_of:
                parent_of[rel_norm] = parent_norm
                children[parent_norm].append(rel_norm)

    if not ks:
        return None, suggested
    root_norm = normalize_keyword(ks[0]["keyword"])
    for k in ks:
        norm = normalize_keyword(k["keyword"])
        if norm != root_norm and norm not in parent_of:
            parent_of[norm] = root_norm
            children[root_norm].append(norm)

    def build_node(norm: str, seen: set) -> Dict[str, Any]:
        k = searched_index.get(norm)
        node = {
            "keyword": labels.get(norm, norm),
            "searched": k is not None,
            "order": k["order"] if k else None,
            "is_hot": k["is_hot"] if k else None,
            "article_count": k["article_count"] if k else 0,
            "super_hot_count": k["super_hot_count"] if k else 0,
            "children": [],
        }
        if k is not None and norm not in seen:
            seen.add(norm)
            node["children"] = [build_node(c, seen) for c in children.get(norm, [])]
        return node

    return build_node(root_norm, set()), suggested


def build_payload(view_id: str, loaded: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    tagged: List[Tuple[int, Dict[str, Any]]] = []
    for ri, (run_id, log) in enumerate(loaded):
        s_list = [s for s in (log.get("searches") or []) if isinstance(s, dict)]
        root = str(log.get("root_keyword") or (s_list[0].get("keyword") if s_list else "") or "")
        started = log.get("timestamp")
        started_iso = None
        if isinstance(started, (int, float)):
            started_iso = datetime.fromtimestamp(float(started), tz=timezone.utc).isoformat()
        runs.append({
            "run_id": run_id,
            "root_keyword": root,
            "started_at": started_iso,
            "file": os.path.basename(run_path(run_id)),
            "searches": len(s_list),
            "quota_used": log.get("quota_used"),
        })
        tagged.extend((ri, s) for s in s_list)

    articles: List[Dict[str, Any]] = []
    by_url: Dict[str, Dict[str, Any]] = {}
    keywords: List[Dict[str, Any]] = []
    super_hot: List[Dict[str, Any]] = []
    source_counter: Counter = Counter()
    removed_total = 0

    for order, (ri, s) in enumerate(tagged):
        kw = str(s.get("keyword") or "").strip()
        decision = s.get("llm_decision") if isinstance(s.get("llm_decision"), dict) else {}
        raw_articles = [a for a in (s.get("articles") or []) if isinstance(a, dict)]
        kept_articles = [a for a in raw_articles if not is_placeholder(a)]
        removed_total += len(raw_articles) - len(kept_articles)
        raw_articles = kept_articles
        reason = str(decision.get("reason") or "").strip()
        related = []
        for r in decision.get("related_keywords") or []:
            if isinstance(r, str) and r.strip() and normalize_keyword(r) not in {normalize_keyword(x) for x in related}:
                related.append(r.strip())

        for i, a in enumerate(raw_articles):
            url = a.get("url") or f"missing-url:{order}:{i}"
            rec = by_url.get(url)
            if rec is None:
                rec = {
                    "id": len(articles),
                    "title": (a.get("title") or "").strip() or "(untitled)",
                    "source": (a.get("source") or "").strip() or "Unknown",
                    "author": a.get("author"),
                    "description": a.get("description") or "",
                    "content": a.get("content") or "",
                    "url": a.get("url"),
                    "published_at": a.get("publishedAt"),
                    "keywords": [],
                    "runs": [],
                    "super_hot": [],
                }
                by_url[url] = rec
                articles.append(rec)
                source_counter[rec["source"]] += 1
            if kw and kw not in rec["keywords"]:
                rec["keywords"].append(kw)
            if ri not in rec["runs"]:
                rec["runs"].append(ri)

        batch_hot = [h for h in (s.get("super_hot_articles") or []) if isinstance(h, dict)]
        for h in batch_hot:
            rec = by_url.get(h.get("url"))
            if rec is None:
                continue
            entry = {
                "keyword": kw,
                "run": ri,
                "reason": str(h.get("super_hot_reason") or "").strip(),
                "rank": h.get("super_hot_rank"),
                "search_order": order,
            }
            rec["super_hot"].append(entry)
            super_hot.append({**entry, "article_id": rec["id"]})

        keywords.append({
            "order": order,
            "run": ri,
            "keyword": kw,
            "article_count": len(raw_articles),
            "is_hot": bool(decision.get("is_hot")),
            "reason": reason,
            "related_keywords": related,
            "super_hot_count": len(batch_hot),
            "llm_failed": reason.startswith("Local LLM"),
        })

    labels: Dict[str, str] = {normalize_keyword(k["keyword"]): k["keyword"] for k in keywords}
    trees: List[Dict[str, Any]] = []
    suggested: set = set()
    for ri in range(len(runs)):
        tree, sug = build_tree([k for k in keywords if k["run"] == ri], labels)
        suggested |= sug
        if tree is not None:
            trees.append({"run": ri, "tree": tree})
    searched_norms = {normalize_keyword(k["keyword"]) for k in keywords}
    unsearched = sorted({labels[n] for n in suggested if n not in searched_norms}, key=str.lower)

    day_counter: Counter = Counter()
    hour_counter: Counter = Counter()
    day_hot: Counter = Counter()
    hour_hot: Counter = Counter()
    times: List[datetime] = []
    for rec in articles:
        dt = parse_iso(rec["published_at"])
        if dt is None:
            continue
        times.append(dt)
        d, h = dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%dT%H:00")
        day_counter[d] += 1
        hour_counter[h] += 1
        if rec["super_hot"]:
            day_hot[d] += 1
            hour_hot[h] += 1

    by_day: List[Dict[str, Any]] = []
    by_hour: List[Dict[str, Any]] = []
    if times:
        lo, hi = min(times), max(times)
        by_day = fill_buckets(day_counter, day_hot,
                              lo.replace(hour=0, minute=0, second=0, microsecond=0),
                              hi, timedelta(days=1), "%Y-%m-%d")
        if (hi - lo) <= timedelta(days=MAX_HOURLY_SPAN_DAYS):
            by_hour = fill_buckets(hour_counter, hour_hot,
                                   lo.replace(minute=0, second=0, microsecond=0),
                                   hi, timedelta(hours=1), "%Y-%m-%dT%H:00")

    top_sources = [{"source": name, "count": n} for name, n in source_counter.most_common(TOP_SOURCES)]
    rest = sum(source_counter.values()) - sum(x["count"] for x in top_sources)
    if rest > 0:
        top_sources.append({"source": "Other", "count": rest, "other_count": len(source_counter) - TOP_SOURCES})

    super_hot.sort(key=lambda h: (h["search_order"], h["rank"] if isinstance(h["rank"], int) else 999))

    totals = {
        "runs": len(runs),
        "searches": len(keywords),
        "quota_used": sum(r["quota_used"] for r in runs if isinstance(r["quota_used"], int)),
        "article_hits": sum(k["article_count"] for k in keywords),
        "unique_articles": len(articles),
        "super_hot_flags": len(super_hot),
        "super_hot_unique": len({h["article_id"] for h in super_hot}),
        "unique_sources": len(source_counter),
        "removed_articles": removed_total,
        "hot_keywords": sum(1 for k in keywords if k["is_hot"]),
        "llm_failures": sum(1 for k in keywords if k["llm_failed"]),
        "suggested_keywords": len(suggested),
        "unsearched_keywords": len(unsearched),
        "earliest_published": min(times).isoformat() if times else None,
        "latest_published": max(times).isoformat() if times else None,
    }

    started_values = [r["started_at"] for r in runs if r["started_at"]]
    return {
        "view_id": view_id,
        "run_id": view_id,
        "runs": runs,
        "root_keyword": " + ".join(r["root_keyword"] for r in runs),
        "started_at": min(started_values) if started_values else None,
        "totals": totals,
        "keywords": keywords,
        "trees": trees,
        "unsearched_keywords": unsearched,
        "sources": top_sources,
        "timeline": {"by_day": by_day, "by_hour": by_hour},
        "super_hot": super_hot,
        "articles": articles,
    }


def run_listing_entry(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    t = payload["totals"]
    return {
        "run_id": run_id,
        "file": payload["runs"][0]["file"],
        "root_keyword": payload["root_keyword"],
        "started_at": payload["started_at"],
        "searches": t["searches"],
        "unique_articles": t["unique_articles"],
        "super_hot": t["super_hot_unique"],
        "has_summary": load_summary(run_id) is not None,
    }


SYSTEM_PROMPT = """You are a news analyst assistant embedded in a local dashboard for an automated news-discovery agent. The agent started from a root keyword, searched NewsAPI, and used a local LLM to judge each keyword's batch of articles, suggest follow-up keywords and flag "super-hot" stories.

Answer the user's questions using the run data below. Ground every claim in that data and name the article titles and sources you rely on. If the data does not contain the answer, say so plainly instead of guessing. Keep answers concise; use markdown lists or short headings when they help."""

SUMMARY_PROMPT = """Write a briefing on this news-discovery run for someone who has not seen it. Use these markdown headings:

## Headline
Two or three sentences on what the run found overall.

## Key themes
Bulleted list of the main themes across the articles.

## Stories that matter most
The super-hot stories that deserve attention and why, naming titles and sources.

## Watch next
Which suggested keywords that were not searched look most promising, and why.

Keep it under 400 words and rely only on the run data provided."""


def snippet(article: Dict[str, Any], limit: int) -> str:
    text = (article.get("description") or article.get("content") or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit] + ("..." if len(text) > limit else "")


def build_context(payload: Dict[str, Any], budget: int, focus_ids: List[int]) -> Tuple[str, Dict[str, Any]]:
    t = payload["totals"]
    arts = payload["articles"]
    parts: List[str] = []

    multi = len(payload["runs"]) > 1
    if multi:
        run_lines = [f'- "{r["root_keyword"]}" started {r["started_at"]} (UTC), {r["searches"]} searches' for r in payload["runs"]]
        overview_head = "Runs combined in this view:\n" + "\n".join(run_lines) + "\n"
    else:
        overview_head = f"Root keyword: {payload['root_keyword']}\nRun started (UTC): {payload['started_at']}\n"
    parts.append(
        "RUN OVERVIEW\n"
        + overview_head
        + f"Searches: {t['searches']} | Unique articles: {t['unique_articles']} | "
        f"Super-hot stories: {t['super_hot_unique']} | Sources: {t['unique_sources']} | "
        f"Keywords judged hot: {t['hot_keywords']} of {t['searches']}\n"
        f"Suggested but not searched: {', '.join(payload['unsearched_keywords']) or 'none'}"
    )

    summary = payload.get("summary")
    if summary:
        parts.append("PREVIOUSLY GENERATED BRIEFING\n" + summary["text"][:3000])

    kw_lines = ["KEYWORD DECISIONS (in search order):"]
    for k in payload["keywords"]:
        verdict = "HOT" if k["is_hot"] else "not hot"
        nxt = ", ".join(k["related_keywords"]) or "none"
        run_tag = f' (run "{payload["runs"][k["run"]]["root_keyword"]}")' if multi else ""
        kw_lines.append(
            f'{k["order"] + 1}. "{k["keyword"]}"{run_tag} [{verdict}] {k["article_count"]} articles, '
            f'{k["super_hot_count"]} super-hot. Reason: {k["reason"] or "n/a"} Suggested next: {nxt}'
        )
    parts.append("\n".join(kw_lines))

    hot = payload["super_hot"][:MAX_SUPER_HOT_IN_CONTEXT]
    sh_lines = ["SUPER-HOT STORIES:"]
    for h in hot:
        a = arts[h["article_id"]]
        sh_lines.append(
            f'- "{a["title"]}" ({a["source"]}, {a["published_at"]}) found via "{h["keyword"]}". '
            f'Why: {h["reason"] or "n/a"} Snippet: {snippet(a, 160)}'
        )
    if len(payload["super_hot"]) > len(hot):
        sh_lines.append(f"... {len(payload['super_hot']) - len(hot)} more super-hot flags omitted.")
    parts.append("\n".join(sh_lines))

    focused = 0
    if focus_ids:
        f_lines = ["ARTICLES THE USER IS FOCUSING ON (full available text):"]
        for aid in focus_ids[:MAX_FOCUS_ARTICLES]:
            if not isinstance(aid, int) or aid < 0 or aid >= len(arts):
                continue
            a = arts[aid]
            focused += 1
            f_lines.append(
                f'- "{a["title"]}" ({a["source"]}, {a["published_at"]}, by {a["author"] or "unknown"}) '
                f'keywords: {", ".join(a["keywords"])}\n  Description: {a["description"] or "n/a"}\n'
                f'  Content: {a["content"] or "n/a"}\n  URL: {a["url"]}'
            )
        if focused:
            parts.append("\n".join(f_lines))

    fixed = "\n\n".join(parts)
    remaining = budget - len(fixed) - 200
    head_lines = ["ALL HEADLINES (title | source | published | found via keywords):"]
    used, shown = 0, 0
    for a in arts:
        line = f'- {a["title"]} | {a["source"]} | {a["published_at"]} | {", ".join(a["keywords"])}'
        if used + len(line) + 1 > remaining:
            break
        head_lines.append(line)
        used += len(line) + 1
        shown += 1
    if shown < len(arts):
        head_lines.append(f"... {len(arts) - shown} more headlines omitted to fit the context window.")
    parts.append("\n".join(head_lines))

    context = "\n\n".join(parts)
    meta = {
        "keywords": len(payload["keywords"]),
        "super_hot_included": len(hot),
        "super_hot_total": len(payload["super_hot"]),
        "headlines_included": shown,
        "headlines_total": len(arts),
        "focused_articles": focused,
        "has_summary": bool(summary),
        "context_chars": len(context),
    }
    return context, meta


def probe_ollama() -> Tuple[bool, List[str], Optional[str]]:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
    except requests.RequestException as e:
        return False, [], str(e)
    if resp.status_code != 200:
        return False, [], f"HTTP {resp.status_code}"
    try:
        models = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
    except ValueError:
        models = []
    return True, models, None


def stream_ollama_chat(model: str, messages: List[Dict[str, str]], meta: Dict[str, Any],
                       on_complete: Optional[Callable[[str], None]] = None) -> Response:

    def generate():
        yield json.dumps({"meta": {**meta, "model": model, "num_ctx": OLLAMA_NUM_CTX}}) + "\n"
        collected: List[str] = []
        finished = False
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": OLLAMA_NUM_CTX},
        }
        try:
            with requests.post(f"{OLLAMA_HOST}/api/chat", json=body, stream=True,
                               timeout=(10, CHAT_TIMEOUT)) as resp:
                if resp.status_code != 200:
                    yield json.dumps({"error": f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"}) + "\n"
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        js = json.loads(line)
                    except ValueError:
                        continue
                    if js.get("error"):
                        yield json.dumps({"error": f"Ollama error: {js['error']}"}) + "\n"
                        return
                    piece = (js.get("message") or {}).get("content") or ""
                    if piece:
                        collected.append(piece)
                        yield json.dumps({"content": piece}) + "\n"
                    if js.get("done"):
                        finished = True
                        yield json.dumps({
                            "done": True,
                            "eval_count": js.get("eval_count"),
                            "prompt_eval_count": js.get("prompt_eval_count"),
                            "total_duration": js.get("total_duration"),
                        }) + "\n"
                        return
        except requests.RequestException as e:
            yield json.dumps({"error": f"Could not reach Ollama at {OLLAMA_HOST}: {e}"}) + "\n"
        finally:
            text = "".join(collected).strip()
            if finished and text and on_complete:
                try:
                    on_complete(text)
                except Exception as e:
                    logger.warning(f"on_complete failed: {e}")

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def clean_history(raw: Any) -> List[Dict[str, str]]:
    out = []
    for m in raw if isinstance(raw, list) else []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content.strip()})
    return out[-MAX_HISTORY_MESSAGES:]


@app.get("/")
def index():
    return render_template("index.html", config={
        "data_dir": DATA_DIR,
        "ollama_host": OLLAMA_HOST,
        "default_model": OLLAMA_MODEL,
    })


@app.get("/api/status")
def api_status():
    reachable, models, err = probe_ollama()
    return jsonify({
        "data_dir": DATA_DIR,
        "run_count": len(discover_runs()),
        "ollama": {
            "host": OLLAMA_HOST,
            "default_model": OLLAMA_MODEL,
            "reachable": reachable,
            "models": models,
            "error": err,
            "num_ctx": OLLAMA_NUM_CTX,
            "context_chars": CHAT_CONTEXT_CHARS,
        },
    })


@app.get("/api/runs")
def api_runs():
    runs = []
    for run_id in discover_runs():
        payload = load_payload(run_id)
        if payload is not None:
            runs.append(run_listing_entry(run_id, payload))
    return jsonify({"data_dir": DATA_DIR, "runs": runs})


@app.get("/api/runs/<run_id>")
def api_run(run_id: str):
    payload = load_payload(run_id)
    if payload is None:
        abort(404, description="Run not found or unreadable")
    return jsonify({**payload, "summary": load_summary(payload["view_id"])})


@app.post("/api/runs/<run_id>/summary")
def api_run_summary(run_id: str):
    payload = load_payload(run_id)
    if payload is None:
        abort(404, description="Run not found or unreadable")
    body = request.get_json(silent=True) or {}
    model = str(body.get("model") or OLLAMA_MODEL).strip() or OLLAMA_MODEL
    context, meta = build_context({**payload, "summary": None}, CHAT_CONTEXT_CHARS, [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context},
        {"role": "user", "content": SUMMARY_PROMPT},
    ]
    view_id = payload["view_id"]
    return stream_ollama_chat(model, messages, meta, on_complete=lambda text: save_summary(view_id, model, text))


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    run_id = str(body.get("run_id") or "")
    payload = load_payload(run_id)
    if payload is None:
        abort(404, description="Run not found or unreadable")
    history = clean_history(body.get("messages"))
    if not history or history[-1]["role"] != "user":
        abort(400, description="The last message must come from the user")
    model = str(body.get("model") or OLLAMA_MODEL).strip() or OLLAMA_MODEL
    focus_ids = [i for i in (body.get("focus_ids") or []) if isinstance(i, int)]
    context, meta = build_context({**payload, "summary": load_summary(payload["view_id"])}, CHAT_CONTEXT_CHARS, focus_ids)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context}] + history
    return stream_ollama_chat(model, messages, meta)


@app.errorhandler(404)
def not_found(err):
    if request.path.startswith("/api/"):
        return jsonify({"error": getattr(err, "description", "Not found")}), 404
    return err


@app.errorhandler(400)
def bad_request(err):
    return jsonify({"error": getattr(err, "description", "Bad request")}), 400


if __name__ == "__main__":
    logger.info(f"Data directory: {DATA_DIR} ({len(discover_runs())} run file(s) found)")
    logger.info(f"Ollama: {OLLAMA_HOST} (default model: {OLLAMA_MODEL}, num_ctx: {OLLAMA_NUM_CTX})")
    logger.info(f"Dashboard: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, threaded=True)
