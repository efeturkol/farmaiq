import pandas as pd
from collections import Counter

# ─── Prospektüs yan etki listeleri (PDF'lerden çıkarıldı) ───

prospektus = {
    "apranax": [
        "baş dönmesi", "rehavet", "baş ağrısı", "konsantrasyon bozukluğu",
        "görme bulanıklığı", "kulak çınlaması", "palpitasyon", "ödem",
        "konjestif kalp yetmezliği", "hipertansiyon", "dispne",
        "peptik ülser", "mide yanması", "bulantı", "kusma", "diyare",
        "konstipasyon", "karın ağrısı", "kaşıntı", "deri döküntüsü",
        "terleme", "saç dökülmesi", "böbrek rahatsızlıkları",
        "depresyon", "uykusuzluk", "konfüzyon", "hemolitik anemi"
    ],
    "paxil": [
        "bulantı", "konstipasyon", "diyare", "kusma", "ağız kuruluğu",
        "baş ağrısı", "sersemlik", "titreme", "uyuklama", "uykusuzluk",
        "ajitasyon", "anormal rüyalar", "konfüzyon", "halüsinasyon",
        "bulanık görme", "sinüs taşikardisi", "terleme", "deri döküntüsü",
        "cinsel disfonksiyon", "güçsüzlük", "kilo artışı", "iştah azalması",
        "karaciğer enzim artışı", "trombositopeni", "hiponatremi"
    ],
    "klaritromisin": [
        "ishal", "kusma", "karın ağrısı", "hazımsızlık", "bulantı",
        "uykusuzluk", "baş dönmesi", "kulak çınlaması", "titreme",
        "halsizlik", "ağız kuruluğu", "kabızlık", "karaciğer enzim artışı",
        "kötü rüyalar", "tat alma bozukluğu", "çarpıntı", "vertigo",
        "duyma bozukluğu", "iştahsızlık", "lökopeni", "eozinofili"
    ],
    "nexium": [
        "baş ağrısı", "ishal", "karın ağrısı", "kabızlık", "gaz",
        "bulantı", "kusma", "şişkinlik", "ağız kuruluğu",
        "sersemlik", "uykusuzluk", "baş dönmesi", "karaciğer enzim artışı",
        "deri döküntüsü", "kaşıntı", "artralji", "miyalji",
        "yorgunluk", "terleme", "saç dökülmesi", "görme bozukluğu",
        "depresyon", "ajitasyon", "halüsinasyon", "trombositopeni",
        "hiponatremi", "bronkospazm", "kemik kırığı riski"
    ],
    "cipram": [
        "uykulu hissetme", "uykusuzluk", "terleme", "ağız kuruluğu",
        "bulantı", "iştah azalması", "ajitasyon", "cinsel dürtü azalması",
        "titreme", "baş dönmesi", "ishal", "kusma", "kabızlık",
        "kaşıntı", "yorgunluk", "kilo azalması", "anormal rüyalar",
        "sinirlilik", "konfüzyon", "halüsinasyon", "mani",
        "saç dökülmesi", "deri döküntüsü", "kasılma nöbeti",
        "tat alma bozukluğu", "karaciğer fonksiyon bozukluğu",
        "hiponatremi", "anormal kalp ritmi"
    ],
    "prozac": [
        "bulantı", "ishal", "mide bozukluğu", "kusma", "hazımsızlık",
        "ağız kuruluğu", "terleme", "titreme", "baş ağrısı",
        "sersemlik", "uyku sorunları", "huzursuzluk", "yorgunluk",
        "iştah kaybı", "kilo kaybı", "cinsel performans sorunları",
        "halüsinasyon", "konfüzyon", "endişe", "sinirlilik",
        "havale", "karaciğer fonksiyon anormalliği", "sarılık",
        "saç dökülmesi", "bulanık görme", "eklem ağrısı", "kas ağrısı",
        "düşük tansiyon", "çarpıntı", "güneş hassasiyeti"
    ]
}

# ─── Sosyal medya yorumlarından tespit edilen yan etkiler ───

sosyal_medya_anahtar = {
    "sindirim": ["mide", "bulantı", "kusma", "ishal", "mide yanması",
                 "karın ağrısı", "hazımsızlık", "iştah", "metalik tat",
                 "ağız tadı", "şişkinlik", "kramp", "reflü"],
    "norolojik": ["baş dönmesi", "baş ağrısı", "uyku", "halsizlik",
                  "titreme", "sersemlik", "konsantrasyon", "sinirlilik",
                  "kişilik değişikliği", "uykusuzluk", "konfüzyon"],
    "diger": ["libido", "cinsel", "kilo", "iştah", "bağımlılık",
              "karaciğer", "ağız kuruluğu", "terleme"],
    "kardiyovaskuler": ["kalp", "tansiyon", "çarpıntı"],
    "dermatolojik": ["döküntü", "kaşıntı", "cilt", "saç dökülmesi"]
}

df = pd.read_csv("data/labeled/etiketlenmis.csv")
df = df[df["yan_etki"] == 1]

# Her ilaç için sosyal medyada geçen yan etki kategorilerini say
print("=" * 60)
print("SOSYAL MEDYA vs PROSPEKTÜS KARŞILAŞTIRMA ANALİZİ")
print("=" * 60)

ilac_sosyal = {}
for ilac in prospektus.keys():
    ilac_df = df[df["ilac"] == ilac]
    kategori_sayisi = Counter(ilac_df["kategori"].tolist())
    ilac_sosyal[ilac] = kategori_sayisi

# Genel kategori dağılımı
print("\n📊 Sosyal Medyada Yan Etki Kategorisi Dağılımı (ilaç bazında):\n")
print(f"{'İlaç':<20} {'Sindirim':>10} {'Nörolojik':>12} {'Diger':>8} {'Kardio':>8} {'Dermato':>10}")
print("-" * 72)
for ilac, sayilar in ilac_sosyal.items():
    print(f"{ilac:<20} {sayilar.get('sindirim',0):>10} {sayilar.get('norolojik',0):>12} "
          f"{sayilar.get('diger',0):>8} {sayilar.get('kardiyovaskuler',0):>8} "
          f"{sayilar.get('dermatolojik',0):>10}")

# Prospektüs karşılaştırması
print("\n\n📋 PROSPEKTÜS KARŞILAŞTIRMASI:\n")
for ilac in prospektus.keys():
    prospektus_yan_etkiler = set(prospektus[ilac])
    ilac_df = df[df["ilac"] == ilac]
    
    # Sosyal medyada bu ilaca ait yorumlar
    yorumlar = " ".join(ilac_df["yorum"].tolist()).lower()
    
    # Prospektüste olan ama sosyal medyada hiç geçmeyen
    prospektuste_var_sosyalde_yok = []
    for etki in prospektus_yan_etkiler:
        if etki not in yorumlar:
            prospektuste_var_sosyalde_yok.append(etki)
    
    print(f"\n🔵 {ilac.upper()}")
    print(f"   Prospektüste {len(prospektus_yan_etkiler)} yan etki kayıtlı")
    print(f"   Sosyal medya yorum sayısı (yan etki içeren): {len(ilac_df)}")
    if prospektuste_var_sosyalde_yok:
        print(f"   Prospektüste var ama sosyal medyada geçmeyen: "
              f"{', '.join(prospektuste_var_sosyalde_yok[:5])}{'...' if len(prospektuste_var_sosyalde_yok)>5 else ''}")

# En çok raporlanan kategoriler (tüm ilaçlar)
print("\n\n📈 TÜM İLAÇLARDA EN ÇOK RAPORLANAN KATEGORİLER:\n")
genel_kategori = Counter(df["kategori"].tolist())
for kategori, sayi in genel_kategori.most_common():
    bar = "█" * sayi
    print(f"  {kategori:<20} {bar} ({sayi})")

print("\n\n✅ Analiz tamamlandı.")
df_ozet = pd.DataFrame([
    {"ilac": ilac, **sayilar} for ilac, sayilar in ilac_sosyal.items()
]).fillna(0)
df_ozet.to_csv("data/labeled/prospektus_karsilastirma.csv", index=False)
print("[✓] data/labeled/prospektus_karsilastirma.csv kaydedildi")
