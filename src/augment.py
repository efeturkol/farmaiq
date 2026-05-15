import pandas as pd
import random

random.seed(42)

df = pd.read_csv("data/labeled/etiketlenmis.csv")

pozitif = df[df["yan_etki"] == 1].copy()

sinonimler = {
    "mide": ["midemi", "midem", "mide bölgem"],
    "bulantı": ["bulantı hissi", "mide bulantısı", "bulantı yapıyor"],
    "baş dönmesi": ["başım dönüyor", "sersemlik", "baş dönmesi hissettim"],
    "uyku": ["uyku hali", "uyuşukluk", "uyku getiriyor"],
    "yanma": ["yanma hissi", "yanan", "yakar gibi"],
    "iştah": ["iştahımı", "iştah kaybı", "iştahsızlık"],
    "halsizlik": ["halsizlik hissettim", "güçsüzlük", "bitkinlik"],
    "titreme": ["titreme hissettim", "titreşim", "titriyor"],
}

def augment_yorum(yorum):
    yorum_yeni = yorum
    for kelime, alternatifler in sinonimler.items():
        if kelime in yorum_yeni.lower():
            alternatif = random.choice(alternatifler)
            yorum_yeni = yorum_yeni.lower().replace(kelime, alternatif, 1)
            break
    return yorum_yeni

augmented_rows = []
for _, row in pozitif.iterrows():
    yeni_yorum = augment_yorum(row["yorum"])
    if yeni_yorum != row["yorum"]:
        yeni_row = row.copy()
        yeni_row["yorum"] = yeni_yorum
        yeni_row["kaynak"] = "augmented"
        augmented_rows.append(yeni_row)

augmented_df = pd.DataFrame(augmented_rows)
df_final = pd.concat([df, augmented_df], ignore_index=True)

print(f"Orijinal: {len(df)} satır")
print(f"Augmented eklendi: {len(augmented_df)} satır")
print(f"Toplam: {len(df_final)} satır")
print(f"Yan etki var (1): {df_final['yan_etki'].sum()}")
print(f"Yan etki yok (0): {(df_final['yan_etki']==0).sum()}")

df_final.to_csv("data/labeled/etiketlenmis.csv", index=False, encoding="utf-8-sig")
print("[✓] data/labeled/etiketlenmis.csv güncellendi")
