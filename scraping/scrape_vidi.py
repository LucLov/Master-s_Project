import time
import json
import os
import re
import random
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException

from bs4 import BeautifulSoup


START_URL = "https://www.vidi.hr"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "vidi_articles.json")
DEBUG_DIR = os.path.join(BASE_DIR, "debug")
COMMENT_FOOTER = (
    "Za mogućnost komentiranja morate biti prijavljeni.\nAko još nemate korisnički račun registrirajte se ovdje »"
)


ARTICLES_TO_SCRAPE = 50  # povećaj po potrebi

CANDIDATE_SELECTORS = [
    "article",
    "main article",
    "[itemprop='articleBody']",
    "div[itemprop='articleBody']",
    "main",
    ".article",
    ".content",
    ".article-content",
    ".post-content",
    ".entry-content",
]

JUNK_HINTS = [
    "podijeli", "share", "sponzor", "oglas", "newsletter", "pretplat",
    "najčitanije", "vezani", "preporuč", "komentari", "komentar",
    "prijavi", "registr", "cookies", "kolačić", "promo",
]


def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,2000")
    # Stabilniji UA često smanji blokiranja/reset
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def restart_driver(driver):
    try:
        driver.quit()
    except Exception:
        pass
    return get_driver()


def safe_get(driver, url, attempts=3):
    """
    Siguran driver.get s retry/backoff. Vraća True/False.
    """
    for i in range(attempts):
        try:
            driver.get(url)
            return True
        except (ConnectionResetError, WebDriverException) as e:
            wait = 2 * (i + 1)
            print(f"   ⚠ GET failed ({type(e).__name__}: {e}). retry in {wait}s...")
            time.sleep(wait)
    return False


def wait_page_ready(driver, timeout=12):
    WebDriverWait(driver, timeout).until(lambda d: d.execute_script("return document.readyState") == "complete")
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
    except Exception:
        pass


def get_overflow(driver, el):
    try:
        return int(driver.execute_script("return arguments[0].scrollHeight - arguments[0].clientHeight;", el) or 0)
    except Exception:
        return 0


def get_text_len(driver, el):
    try:
        return len((el.text or "").strip())
    except Exception:
        return 0


def get_p_count(driver, el):
    try:
        return int(driver.execute_script("return arguments[0].querySelectorAll('p').length;", el) or 0)
    except Exception:
        return 0


def score_element(driver, el):
    p = get_p_count(driver, el)
    t = get_text_len(driver, el)
    o = get_overflow(driver, el)
    return (p * 10000) + t + (o * 2)


def is_quality_ok(text: str):
    if not text:
        return False
    if len(text) < 600:
        return False
    low = text.lower()
    junk_hits = sum(1 for w in JUNK_HINTS if w in low)
    if junk_hits >= 6:
        return False
    return True


def save_debug(driver, url, reason, html=None):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", urlparse(url).path.strip("/"))[:80] or "unknown"
    fname = f"{safe}__{reason}.html"
    path = os.path.join(DEBUG_DIR, fname)
    try:
        if html is None:
            html = driver.page_source
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"   🧪 Debug saved: {path}")
    except Exception as e:
        print(f"   ⚠ Could not save debug: {e}")


def remove_junk_tags(soup: BeautifulSoup):
    for tag in soup(["nav", "footer", "header", "aside", "script", "style", "noscript"]):
        tag.decompose()

    # VAŽNO: snapshot liste prije decompose da ne ruši iterator
    for el in list(soup.find_all(True)):
        if el is None:
            continue

        # guard: bs4 edge-case (attrs može biti None nakon decompose u istom tree-u)
        if not hasattr(el, "attrs") or el.attrs is None:
            continue

        try:
            classes = el.attrs.get("class") or []
            if isinstance(classes, str):
                classes = [classes]

            cls = " ".join(classes).lower()
            _id = (el.attrs.get("id") or "").lower()
            hay = f"{cls} {_id}"

            if any(k in hay for k in ["ad", "ads", "banner", "promo", "related", "share", "newsletter", "recommend", "sponsor"]):
                el.decompose()
        except Exception:
            # ako se neki tag raspadne, samo ga preskoči
            continue


def extract_text_from_html(container_html: str) -> str:
    soup = BeautifulSoup(container_html, "html.parser")
    remove_junk_tags(soup)

    paragraphs = []
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if len(txt) < 40:
            continue
        low = txt.lower()
        if any(j in low for j in ["vidi više", "foto:", "podijeli", "vezani sadržaji", "najčitanije"]):
            continue
        paragraphs.append(txt)

    if not paragraphs:
        return soup.get_text("\n", strip=True)

    text = "\n".join(paragraphs[:80])

    # ukloni footer za komentiranje (ako postoji)
    if COMMENT_FOOTER in text:
        text = text.split(COMMENT_FOOTER)[0].strip()

    return text



def scroll_page_until_stable(driver, max_rounds=12, pause=0.4):
    last_h = 0
    for _ in range(max_rounds):
        h = driver.execute_script("return document.body.scrollHeight;")
        if h == last_h:
            break
        last_h = h
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.2)


def scroll_container_until_underflow(driver, el, max_rounds=18, pause=0.35):
    overflow = get_overflow(driver, el)
    if overflow <= 0:
        return

    last_top = -1
    last_overflow = overflow

    for _ in range(max_rounds):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", el)
        time.sleep(pause)

        top = driver.execute_script("return arguments[0].scrollTop;", el)
        overflow = get_overflow(driver, el)

        if top == last_top and overflow == last_overflow:
            break

        last_top = top
        last_overflow = overflow


def find_best_container(driver):
    best = None
    best_score = -1

    # 1) selector-based
    for sel in CANDIDATE_SELECTORS:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            for el in els:
                if get_text_len(driver, el) < 200:
                    continue
                s = score_element(driver, el)
                if s > best_score:
                    best = el
                    best_score = s
        except Exception:
            continue

    if best is not None and best_score > 0:
        return best

    # 2) heuristic fallback
    divs = driver.find_elements(By.TAG_NAME, "div")
    for el in divs:
        try:
            if get_text_len(driver, el) < 300:
                continue
            s = score_element(driver, el)
            if s > best_score:
                best = el
                best_score = s
        except Exception:
            continue

    return best


def extract_article_content(driver, url):
    ok = safe_get(driver, url, attempts=3)
    if not ok:
        return ""

    wait_page_ready(driver)

    # scroll page (lazy-load)
    scroll_page_until_stable(driver)

    container = find_best_container(driver)
    if not container:
        save_debug(driver, url, "no_container")
        return ""

    # dynamic load inside container (overflow/underflow)
    scroll_container_until_underflow(driver, container)

    container_html = container.get_attribute("innerHTML") or ""
    content = extract_text_from_html(container_html)

    if not is_quality_ok(content):
        # fallback: pokušaj iz cijelog HTML-a
        full = extract_text_from_html(driver.page_source)
        if is_quality_ok(full):
            return full

        save_debug(driver, url, "quality_fail", html=container_html[:200000])
        return ""

    return content


def scrape_vidi():
    driver = get_driver()

    ok = safe_get(driver, START_URL, attempts=3)
    if not ok:
        print("❌ Could not load START_URL after retries.")
        driver.quit()
        return

    wait_page_ready(driver)
    time.sleep(1.0)

    elems = driver.find_elements(By.TAG_NAME, "a")

    links = []
    for e in elems:
        href = e.get_attribute("href")
        if href and href.startswith("https://www.vidi.hr") and href.count("/") > 4:
            if "/layout/set/poll/" in href or "/content/collectedinfo/" in href:
                continue
            links.append(href)


    links = list(set(links))
    print(f"[INFO] Found {len(links)} article-like links.")

    articles = []
    count = 0
    failures = 0

    for url in links:
        #if count >= ARTICLES_TO_SCRAPE:
        #    break

        print(f"[SCRAPING] {url}")

        try:
            # malo uspori da smanji reset
            time.sleep(random.uniform(0.6, 1.6))

            # title
            ok = safe_get(driver, url, attempts=3)
            if not ok:
                print("   ⚠ Skipped (failed to load after retries)")
                save_debug(driver, url, "load_fail")
                failures += 1
                if failures >= 5:
                    driver = restart_driver(driver)
                    failures = 0
                continue

            wait_page_ready(driver)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            title_tag = soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else ""

            if not title or not title.strip():
                print("   ⚠ Skipped (missing title)")
                save_debug(driver, url, "missing_title")
                failures += 1
                if failures >= 5:
                    driver = restart_driver(driver)
                    failures = 0
                continue


            # content (ponovno get+scroll unutar extract)
            content = extract_article_content(driver, url)

            if len(content) < 80:
                print("   ⚠ Skipped (content too short / quality fail)")
                failures += 1
                if failures >= 5:
                    driver = restart_driver(driver)
                    failures = 0
                continue

            articles.append({
                "url": url,
                "title": title,
                "content": content,
                "source": "vidi"
            })

            print("   ✓ Saved content")
            count += 1
            failures = 0

        except Exception as e:
            import traceback
            print(f"   ❌ Error: {e}")
            print(traceback.format_exc())
            save_debug(driver, url, "exception")
            failures += 1

            # ako se zareda par failova, restart driver
            if failures >= 3:
                driver = restart_driver(driver)
                failures = 0

    try:
        driver.quit()
    except Exception:
        pass

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved {len(articles)} articles → {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape_vidi()
