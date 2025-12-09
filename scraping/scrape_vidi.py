import time
import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup


START_URL = "https://www.vidi.hr"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "vidi_articles.json")
ARTICLES_TO_SCRAPE = 10


def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def extract_main_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # 1. Ukloni nepotrebne dijelove prije analize
    for tag in soup(["nav", "footer", "header", "aside", "script", "style"]):
        tag.decompose()

    # 2. Pronađi div koji ima najviše teksta
    candidate_divs = soup.find_all("div")

    best_div = None
    max_text_len = 0

    for div in candidate_divs:
        text = div.get_text(" ", strip=True)
        if len(text) > max_text_len:
            max_text_len = len(text)
            best_div = div

    if not best_div:
        return ""

    # 3. Unutar glavnog diva pokupi p tagove
    paragraphs = best_div.find_all("p")

    cleaned = []
    for p in paragraphs:
        txt = p.get_text(" ", strip=True)

        # Filtriraj smeće
        if len(txt) < 40:
            continue
        if "Vidi više" in txt or "Foto:" in txt:
            continue

        cleaned.append(txt)

    # 4. Ako ništa nije pronađeno, fallback
    if not cleaned:
        fallback_ps = soup.find_all("p")
        cleaned = [
            p.get_text(" ", strip=True)
            for p in fallback_ps
            if len(p.get_text(strip=True)) > 60
        ]

    return "\n".join(cleaned[:50])  # limit za sigurnost
    #return "\n".join(cleaned)



def scrape_vidi():
    driver = get_driver()
    driver.get(START_URL)
    time.sleep(2)

    elems = driver.find_elements(By.TAG_NAME, "a")

    links = []
    for e in elems:
        href = e.get_attribute("href")
        if href and href.startswith("https://www.vidi.hr") and href.count("/") > 4:
            links.append(href)

    links = list(set(links))
    print(f"[INFO] Found {len(links)} article-like links.")

    articles = []
    count = 0

    for url in links:
        if count >= ARTICLES_TO_SCRAPE:
            break

        print(f"[SCRAPING] {url}")
        driver.get(url)
        time.sleep(2)

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else ""

        content = extract_main_text(html)

        if len(content) < 80:
            print("   ⚠ Skipped (content too short)")
            continue

        articles.append({
            "url": url,
            "title": title,
            "content": content,
            "source": "vidi"
        })

        print("   ✓ Saved full content")
        count += 1

    driver.quit()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved {len(articles)} articles → {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape_vidi()
