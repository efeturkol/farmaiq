import pandas as pd

df = pd.read_csv("data/labeled/etiketlenecek.csv")

sindirim = ["mide", "bulantı", "kusma", "ishal", "kabızlık", "karın", "mide ağrı", 
            "hazımsızlık", "iştah", "metalik", "ağız tadı", "yanma", "kramp", "reflü",
            "sindirim", "dispepsi", "gaz", "şişkinlik", "diyare", "konstipasyon"]

norolojik = ["baş dönmesi", "baş ağrısı", "uyku", "halsizlik", "titreme", "sersem",
             "konsantre", "dikkat", "sinirlilik", "kişilik", "uykusuzluk", "uyuşma",
             "vertigo", "kulak çınlama", "bulanık görme", "parestezi", "konfüzyon",
             "ajitasyon", "halüsinasyon", "depresyon", "anksiyete", "yorgunluk"]

dermatolojik = ["döküntü", "kaşıntı", "ürtiker", "cilt", "deri", "kızarıklık", "alerji"]

kardiyovaskuler = ["kalp", "tansiyon", "çarpıntı", "hipertansiyon", "kardiyak"]

diger = ["libido", "cinsel", "ejak", "kilo", "iştah", "bağımlılık", "karaciğer",
         "böbrek", "kan", "alopesi", "terleme", "ağız kuruluğu", "ödem"]

def etiketle(yorum):
    yorum_lower = yorum.lower()
    
    for kelime in sindirim:
        if kelime in yorum_lower:
            return 1, "sindirim"
    for kelime in norolojik:
        if kelime in yorum_lower:
            return 1, "norolojik"
    for kelime in dermatolojik:
        if kelime in yorum_lower:
            return 1, "dermatolojik"
    for kelime in kardiyovaskuler:
        if kelime in yorum_lower:
            return 1, "kardiyovaskuler"
    for kelime in diger:
        if kelime in yorum_lower:
            return 1, "diger"
    return 0, "yok"

df[["yan_etki", "kategori"]] = df["yorum"].apply(
    lambda x: pd.Series(etiketle(str(x)))
)

print(f"Toplam: {len(df)} satır")
print(f"Yan etki var (1): {df['yan_etki'].sum()}")
print(f"Yan etki yok (0): {(df['yan_etki']==0).sum()}")
print(f"\nKategori dağılımı:\n{df['kategori'].value_counts()}")

df.to_csv("data/labeled/etiketlenmis.csv", index=False, encoding="utf-8-sig")
print("\n[✓] data/labeled/etiketlenmis.csv kaydedildi")
