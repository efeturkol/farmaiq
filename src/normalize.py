import pandas as pd

df = pd.read_csv("data/labeled/etiketlenmis.csv")

def normalize(row):
    yorum = str(row["yorum"]).strip()
    if len(yorum) < 40 and row["yan_etki"] == 1:
        return f"Bu ilacı kullandıktan sonra {yorum.lower()} yaşadım."
    elif len(yorum) < 40 and row["yan_etki"] == 0:
        return f"Bu ilacı kullandım, {yorum.lower()}."
    return yorum

df["yorum"] = df.apply(normalize, axis=1)

print(f"Toplam: {len(df)} satır")
print("\nÖrnek dönüşümler:")
print(df[df["kaynak"] == "google_form"]["yorum"].head(5).tolist())

df.to_csv("data/labeled/etiketlenmis.csv", index=False, encoding="utf-8-sig")
print("\n[✓] Kaydedildi")
