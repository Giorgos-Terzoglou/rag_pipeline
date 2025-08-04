import json
import re
import langdetect
from langdetect.lang_detect_exception import LangDetectException

INPUT_PATH = "data/output/collected_data.json"
OUTPUT_PATH = "data/output/collected_data_clean.json"

# Common boilerplate phrases on IMF & similar sites
BOILERPLATE_PATTERNS = [
    r"About\s+Us", r"Research", r"Countries", r"News", r"Publications", r"Events",
    r"Contact\s+Us", r"Legal\s+Information", r"Privacy\s+Policy",
    r"Subscribe", r"Share", r"Follow\s+Us"
]

def is_english(text):
    try:
        lang = langdetect.detect(text[:500])  # Check first 500 chars
        return lang == "en"
    except LangDetectException:
        return False

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove boilerplate patterns
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()

def clean_json():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []
    for entry in data:
        content = entry.get("content", "")
        if not content.strip():
            continue

        content_clean = clean_text(content)
        if is_english(content_clean):
            cleaned_data.append({
                "source": entry.get("source"),
                "content": content_clean
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Cleaned JSON saved to {OUTPUT_PATH}")
    print(f"📝 Original: {len(data)} entries → Cleaned: {len(cleaned_data)} entries")

if __name__ == "__main__":
    clean_json()
