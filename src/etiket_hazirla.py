import pandas as pd
import os

df = pd.read_csv("data/raw/temiz_veri.csv")
df["yan_etki"] = ""
df["kategori"] = ""
os.makedirs("data/labeled", exist_ok=True)
df.to_csv("data/labeled/etiketlenecek.csv", index=False, encoding="utf-8-sig")
print(f"[✓] {len(df)} satır → data/labeled/etiketlenecek.csv")
print("\nEtiket rehberi:")
print("yan_etki: 1 = yan etki var, 0 = yan etki yok")
print("kategori: sindirim / norolojik / dermatolojik / kardiyovaskuler / diger / yok")
