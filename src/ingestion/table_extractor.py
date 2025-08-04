import pandas as pd
from bs4 import BeautifulSoup

def extract_tables_from_html(results):
    table_entries = []
    for item in results:
        if "tables" in item and item["tables"]:
            for html in item["tables"]:
                df_list = pd.read_html(html)
                for df in df_list:
                    table_entries.append({
                        "source": item["url"],
                        "table": df.to_dict(orient="records")
                    })
    return table_entries
