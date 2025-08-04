import os
import requests
import PyPDF2
import pdfplumber
from io import BytesIO
from urllib.parse import urlparse
import hashlib

def safe_filename(url, ext):
    """Short hash-based filename to avoid long names."""
    hash_part = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"{hash_part}.{ext}"

def extract_text_and_tables_from_pdf(url, save_dir="data/pdfs"):
    os.makedirs(save_dir, exist_ok=True)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()

        filename = safe_filename(url, "pdf")
        pdf_path = os.path.join(save_dir, filename)

        with open(pdf_path, "wb") as f:
            f.write(resp.content)

        # Extract text
        text_reader = PyPDF2.PdfReader(BytesIO(resp.content))
        text_content = "\n".join(page.extract_text() or "" for page in text_reader.pages)

        # Extract tables
        tables = []
        with pdfplumber.open(BytesIO(resp.content)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    tables.append(table)

        return pdf_path, text_content, tables
    except Exception as e:
        print(f"PDF processing error for {url}: {e}")
        return None, "", []

def load_pdfs(results, text_dir="data/pdf_texts", tables_dir="data/pdf_tables"):
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    pdf_docs = []
    table_entries = []

    for item in results:
        if item.get("pdf"):
            url = item["url"]
            pdf_path, text, tables = extract_text_and_tables_from_pdf(url)
            if pdf_path:
                text_file = os.path.join(text_dir, safe_filename(url, "txt"))
                with open(text_file, "w", encoding="utf8") as f:
                    f.write(text)
                
                pdf_docs.append({"source": url, "content": text, "pdf_path": pdf_path})

                for idx, table in enumerate(tables):
                    table_file = os.path.join(tables_dir, f"{safe_filename(url, 'table')}_{idx}.csv")
                    with open(table_file, "w", encoding="utf8") as tf:
                        for row in table:
                            tf.write(",".join(str(cell) if cell else "" for cell in row) + "\n")
                    table_entries.append({"source": url, "table": table, "table_path": table_file})
    
    return pdf_docs, table_entries
