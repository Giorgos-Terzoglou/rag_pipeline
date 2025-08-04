import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import csv

def crawl_site(start_url, max_pages=50):
    visited = set()
    results = []

    fail_log_path = "data/output/failed_links.csv"
    os.makedirs("data/output", exist_ok=True)
    with open(fail_log_path, "w", newline="", encoding="utf-8") as fail_file:
        writer = csv.writer(fail_file)
        writer.writerow(["Failed URL", "Reason"])

    def log_failed(url, reason):
        print(f"⚠️ Failed: {url} ({reason})")
        with open(fail_log_path, "a", newline="", encoding="utf-8") as fail_file:
            writer = csv.writer(fail_file)
            writer.writerow([url, reason])

    def crawl(url):
        if url in visited or len(visited) >= max_pages:
            return
        visited.add(url)

        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                log_failed(url, f"HTTP {r.status_code}")
                return

            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ").strip()
            results.append({"url": url, "content": text})

            for link in soup.find_all("a", href=True):
                absolute = urljoin(url, link["href"])
                # Same domain check
                if urlparse(absolute).netloc == urlparse(start_url).netloc:
                    # Always collect PDFs
                    if absolute.lower().endswith(".pdf"):
                        results.append({"url": absolute, "pdf": True})
                    # Crawl HTML pages (detail pages, listings, etc.)
                    elif absolute not in visited:
                        crawl(absolute)

        except Exception as e:
            log_failed(url, str(e))

    crawl(start_url)
    return results
