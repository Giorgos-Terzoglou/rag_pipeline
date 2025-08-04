import os
import json
import csv
import argparse
from src.ingestion import crawl_site, load_pdfs, extract_tables_from_html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    parser.add_argument("--start-url", type=str, required=True, help="Starting URL for crawling")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum number of pages to crawl")
    args = parser.parse_args()

    START_URL = args.start_url
    MAX_PAGES = args.max_pages

    print(f"🚀 Starting ingestion from: {START_URL} (max {MAX_PAGES} pages)")

    os.makedirs("data/output", exist_ok=True)
    
    # CSV logs
    log_path = "data/output/data_log.csv"
    fail_log_path = "data/output/failed_links.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Source URL", "Type", "Local Path", "Table Count"])
    
    with open(fail_log_path, "w", newline="", encoding="utf-8") as fail_file:
        writer = csv.writer(fail_file)
        writer.writerow(["Failed URL", "Reason"])

    # 1️⃣ Crawl
    results = crawl_site(START_URL, max_pages=MAX_PAGES)
    print(f"✅ Crawled {len(results)} pages")

    html_docs = [r for r in results if not r.get("pdf")]

    # 2️⃣ Extract PDFs (returns text + tables)
    pdf_docs, pdf_tables = load_pdfs(results)
    print(f"✅ Extracted {len(pdf_docs)} PDFs")
    print(f"✅ Extracted {len(pdf_tables)} tables from PDFs")

    # Log PDFs and PDF tables
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        for pdf in pdf_docs:
            local_path = f"data/pdf_texts/{os.path.basename(pdf['source'])}.txt"
            writer.writerow([pdf["source"], "PDF Text", local_path, 0])
        for t in pdf_tables:
            local_path = f"data/pdf_tables/{os.path.basename(t['source'])}_table.csv"
            writer.writerow([t["source"], "PDF Table", local_path, len(t["table"])])

    # 3️⃣ Extract tables from HTML
    html_tables = extract_tables_from_html(html_docs)
    print(f"✅ Extracted {len(html_tables)} tables from HTML")

    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        for html in html_docs:
            writer.writerow([html["url"], "HTML Text", "", 0])
        for t in html_tables:
            writer.writerow([t["source"], "HTML Table", "", len(t["table"])])

    # 4️⃣ Combine documents
    all_docs = []
    all_docs += [{"source": r["url"], "content": r["content"]} for r in html_docs]
    all_docs += [{"source": p["source"], "content": p["content"]} for p in pdf_docs]
    all_docs += [{"source": t["source"], "content": "\n".join(str(row) for row in t["table"])} for t in pdf_tables]
    all_docs += [{"source": t["source"], "content": "\n".join(str(row) for row in t["table"])} for t in html_tables]

    print(f"📄 Total collected documents: {len(all_docs)}")

    # 5️⃣ Save combined JSON
    output_path = "data/output/collected_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print(f"💾 Data saved to {output_path}")
    print(f"📜 Provenance log saved to {log_path}")
    print(f"⚠️ Failed links logged to {fail_log_path}")
