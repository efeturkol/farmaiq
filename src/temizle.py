import pandas as pd

df = pd.read_csv("data/raw/ham_veri.csv")
print(f"Ham veri: {len(df)} satır")

# Duplicate temizle
df = df.drop_duplicates(subset=["yorum"])
print(f"Duplicate temizlendi: {len(df)} satır")

# Çok kısa yorumları at (50 karakterden az)
df = df[df["yorum"].str.len() >= 50]
print(f"Kısa yorumlar temizlendi: {len(df)} satır")

# İlaç bazında dağılım
print("\nİlaç bazında dağılım:")
print(df["ilac"].value_counts())

df.to_csv("data/raw/temiz_veri.csv", index=False, encoding="utf-8-sig")
print("\n[✓] data/raw/temiz_veri.csv kaydedildi")
