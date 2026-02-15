import os
import json
import time
from typing import Dict, Any, List, Optional
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename="agent_runtime.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

logger.info("=== Starting News Discovery Agent (Ollama) ===")

NEWS_API_KEY = os.getenv("NEWSAPI_KEY", "")
NEWS_BASE_URL = "https://newsapi.org/v2/everything"
DAILY_QUOTA = int(os.getenv("DAILY_QUOTA", "10"))
ROOT_KEYWORD = os.getenv("ROOT_KEYWORD", "")
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "10"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

logger.info(f"Loaded config: ROOT_KEYWORD={ROOT_KEYWORD}, PAGE_SIZE={PAGE_SIZE}, DAILY_QUOTA={DAILY_QUOTA}")
logger.info(f"Ollama model: {OLLAMA_MODEL} @ {OLLAMA_HOST}")

def newsapi_search(keyword: str) -> Dict[str, Any]:
    logger.info(f"Searching NewsAPI for: {keyword}")

    params = {
        "q": keyword,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": PAGE_SIZE,
        "page": 1,
        "sortBy": "publishedAt",
    }

    logger.debug(f"NewsAPI request params: {params}")

    try:
        resp = requests.get(NEWS_BASE_URL, params=params, timeout=15)
    except Exception as e:
        logger.error(f"NewsAPI request failed for '{keyword}': {e}")
        return {"keyword": keyword, "error": str(e), "articles": []}

    if resp.status_code != 200:
        logger.error(f"NewsAPI returned HTTP {resp.status_code} for keyword '{keyword}'")
        return {"keyword": keyword, "error": f"HTTP {resp.status_code}", "articles": []}

    js = resp.json()
    articles = js.get("articles", [])
    logger.info(f"NewsAPI returned {len(articles)} articles for '{keyword}'")

    normalized = []
    for a in articles:
        normalized.append({
            "title": a.get("title"),
            "source": (a.get("source") or {}).get("name"),
            "author": a.get("author"),
            "description": a.get("description"),
            "content": a.get("content"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
        })

    return {
        "keyword": keyword,
        "articles": normalized,
        "count": len(normalized)
    }

def ollama_generate(prompt: str) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}")

    js = resp.json()
    return (js.get("response") or "").strip()

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    stripped = text.strip()

    try:
        return json.loads(stripped)
    except Exception:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None

def ask_local_llm_for_next_keywords(keyword: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info(f"Sending {len(articles)} articles to local LLM for keyword '{keyword}'")

    articles_json = json.dumps(articles, indent=2)[:60000]

    prompt = f"""
        You are a news-discovery agent. You receive batches of articles from NewsAPI.

        ROOT SEARCH: "{keyword}"

        ARTICLES (truncated JSON follows, zero-based indexing):
        {articles_json}

        TASK:
        1. Analyze whether this keyword is producing new, interesting, or emerging stories.
        2. Suggest 0-3 related keywords worth searching next, if and only if they appear to be hot or emerging topics.
        3. Keep suggestions short (1-3 words each).
        4. From THIS batch of articles, identify 0-5 stories that are "hot hot hot" (super important) relative to the others:
        - Especially time-sensitive, central to the trend, or high-impact.
        - Use the ZERO-BASED index into the ARTICLES list.

        IMPORTANT OUTPUT RULES:
        - Respond with ONLY a valid JSON object.
        - Do NOT include backticks, markdown, or explanatory text outside JSON.

        The JSON MUST have this shape:

        {{
        "is_hot": true/false,
        "reason": "short analysis of this keyword's batch",
        "related_keywords": ["kw1", "kw2"],
        "super_hot_articles": [
            {{
            "index": 0,
            "reason": "why this story is super hot"
            }}
        ]
        }}
    """

    try:
        content_text = ollama_generate(prompt)
    except Exception as e:
        logger.error(f"Local LLM call failed for '{keyword}': {e}")
        return {
            "is_hot": False,
            "reason": f"Local LLM error: {e}",
            "related_keywords": [],
            "super_hot_articles": []
        }

    logger.info(f"Local LLM raw response preview for '{keyword}': {repr(content_text[:300])}")

    parsed = extract_json_object(content_text)
    if not parsed or not isinstance(parsed, dict):
        logger.error(f"Local LLM returned invalid JSON for '{keyword}'")
        return {
            "is_hot": False,
            "reason": "Local LLM returned invalid JSON.",
            "related_keywords": [],
            "super_hot_articles": []
        }

    parsed.setdefault("related_keywords", [])
    parsed.setdefault("super_hot_articles", [])

    logger.info(f"Local LLM decision for '{keyword}': {parsed}")
    return parsed

def save_log(log_data: Dict[str, Any], filename: str):
    logger.info(f"Saving log file: {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

def save_super_hot_timeline(stories: List[Dict[str, Any]], filename: str):
    logger.info(f"Saving super hot timeline file: {filename}")
    payload = {"stories": stories}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def save_all_stories_timeline(stories: List[Dict[str, Any]], filename: str):
    logger.info(f"Saving all stories timeline file: {filename}")
    payload = {"stories": stories}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def run_daily_news_agent(root_keyword: str, quota: int = DAILY_QUOTA):
    logger.info(f"Starting agent run with root keyword: {root_keyword}")

    search_queue = [root_keyword]
    visited: set[str] = set()

    log: Dict[str, Any] = {
        "root_keyword": root_keyword,
        "searches": [],
        "timestamp": time.time(),
        "quota_used": 0,
        "super_hot_articles": []
    }

    while search_queue and log["quota_used"] < quota:
        keyword = search_queue.pop(0)
        logger.info(f"Processing keyword: {keyword}")

        if keyword in visited:
            logger.info(f"Skipping '{keyword}' (already visited)")
            continue
        visited.add(keyword)

        result = newsapi_search(keyword)
        articles = result.get("articles", [])
        logger.info(f"Retrieved {len(articles)} articles for '{keyword}'")

        decision = ask_local_llm_for_next_keywords(keyword, articles)

        super_hot_specs = decision.get("super_hot_articles", []) or []
        batch_super_hot: List[Dict[str, Any]] = []

        for idx, spec in enumerate(super_hot_specs):
            art_index = spec.get("index")
            reason = spec.get("reason", "")

            if not isinstance(art_index, int):
                logger.warning(f"Super-hot spec has non-int index for '{keyword}': {spec}")
                continue

            if art_index < 0 or art_index >= len(articles):
                logger.warning(
                    f"Super-hot index out of range for '{keyword}': {art_index} (len={len(articles)})"
                )
                continue

            base_article = articles[art_index].copy()

            enriched = {
                **base_article,
                "source_keyword": keyword,
                "super_hot_reason": reason,
                "super_hot_rank": idx,
            }

            batch_super_hot.append(enriched)
            log["super_hot_articles"].append(enriched)

        log_entry = {
            "keyword": keyword,
            "article_count": len(articles),
            "articles": articles,
            "llm_decision": decision,
            "super_hot_articles": batch_super_hot,
        }
        log["searches"].append(log_entry)
        log["quota_used"] += 1

        logger.info(f"Quota used: {log['quota_used']}/{quota}")

        next_keywords = decision.get("related_keywords", [])
        logger.info(f"LLM suggests next keywords: {next_keywords}")

        for kw in next_keywords:
            if kw not in visited:
                logger.info(f"Adding '{kw}' to search queue")
                search_queue.append(kw)

        if len(search_queue) > 100:
            logger.warning("Search queue grew too large. Stopping early.")
            break

    ts = int(time.time())

    log_filename = f"news_agent_log_{ts}.json"
    save_log(log, log_filename)
    logger.info(f"=== Agent run complete. Log saved to {log_filename} ===")

    seen_urls: set[str] = set()
    timeline_stories: List[Dict[str, Any]] = []

    for art in log.get("super_hot_articles", []):
        url = art.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        timeline_stories.append(
            {
                "title": art.get("title"),
                "source": art.get("source"),
                "story": art.get("content") or art.get("description") or "",
                "timestamp": art.get("publishedAt"),
                "url": url,
            }
        )

    super_hot_filename = f"news_super_hot_{ts}.json"
    save_super_hot_timeline(timeline_stories, super_hot_filename)
    logger.info(f"Super hot timeline saved to {super_hot_filename}")

    all_seen_urls: set[str] = set()
    all_timeline_stories: List[Dict[str, Any]] = []

    for search in log.get("searches", []):
        for art in search.get("articles", []):
            url = art.get("url")
            if not url or url in all_seen_urls:
                continue
            all_seen_urls.add(url)

            all_timeline_stories.append(
                {
                    "title": art.get("title"),
                    "source": art.get("source"),
                    "story": art.get("content") or art.get("description") or "",
                    "timestamp": art.get("publishedAt"),
                    "url": url,
                }
            )

    all_stories_filename = f"news_all_stories_{ts}.json"
    save_all_stories_timeline(all_timeline_stories, all_stories_filename)
    logger.info(f"All stories timeline saved to {all_stories_filename}")

    logger.info(
        f"=== Agent run complete. "
        f"log={log_filename}, super_hot={super_hot_filename}, all_stories={all_stories_filename} ==="
    )

    return log_filename, super_hot_filename, all_stories_filename

if __name__ == "__main__":
    logger.info("=== Executing agent script (Ollama) ===")

    if not ROOT_KEYWORD:
        logger.error("ROOT_KEYWORD is missing. Please set it in your .env file.")
        raise SystemExit(1)

    if not NEWS_API_KEY:
        logger.error("NEWSAPI_KEY is missing. Please set it in your .env file.")
        raise SystemExit(1)

    try:
        tags = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=10)
        if tags.status_code != 200:
            raise RuntimeError(f"/api/tags returned HTTP {tags.status_code}")
        logger.info("Ollama is reachable.")
    except Exception as e:
        logger.error(f"Ollama is not reachable at {OLLAMA_HOST}: {e}")
        raise SystemExit(1)

    try:
        logger.info(f"Starting daily agent run with root keyword: '{ROOT_KEYWORD}'")
        log_file, super_hot_file, all_stories_file = run_daily_news_agent(ROOT_KEYWORD, DAILY_QUOTA)
        logger.info(
            f"Agent run complete. "
            f"log={log_file}, super_hot={super_hot_file}, all_stories={all_stories_file}"
        )
    except Exception as e:
        logger.exception(f"Agent encountered an unhandled error: {e}")
        raise
