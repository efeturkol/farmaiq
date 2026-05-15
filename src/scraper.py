import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

ILACLAR = [
    "apranax", "majezik", "parol", "voltaren", "arveles",
    "cipralex", "prozac", "sertralin", "paxil", "lustral", "cipram",
    "augmentin", "amoksisilin", "klaritomisin", "klaritromisin", "penicilin",
    "nexium", "lansoprazol", "omeprazol",
    "xanax", "diazepam", "ritalin",
    "paracetamol", "ibuprofen", "naproxen",
]

def get_entries(baslik, max_sayfa=20):
    entries = []
    for sayfa in range(1, max_sayfa + 1):
        url = f"https://eksisozluk.com/{baslik}?p={sayfa}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                break
            soup = BeautifulSoup(r.text, "html.parser")
            items = soup.select("li[data-id] div.content")
            if not items:
                break
            for item in items:
                text = item.get_text(separator=" ").strip()
                if len(text) > 20:
                    entries.append({"ilac": baslik, "yorum": text, "kaynak": "eksisozluk"})
            print(f"[+] {baslik} sayfa {sayfa}: {len(items)} entry")
            time.sleep(2)
        except Exception as e:
            print(f"[!] Hata: {baslik} sayfa {sayfa}: {e}")
            break
    return entries

def main():
    tum_veriler = []
    for ilac in ILACLAR:
        print(f"\n--- {ilac} taranıyor ---")
        veriler = get_entries(ilac, max_sayfa=20)
        tum_veriler.extend(veriler)
        print(f"[✓] {ilac}: toplam {len(veriler)} entry")
        time.sleep(3)

    df = pd.DataFrame(tum_veriler)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/ham_veri.csv", index=False, encoding="utf-8-sig")
    print(f"\n[✓] Toplam {len(df)} entry kaydedildi → data/raw/ham_veri.csv")

if __name__ == "__main__":
    main()
