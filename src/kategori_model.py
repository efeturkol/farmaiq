from sentence_transformers import SentenceTransformer, util
import torch
import pickle

model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

KATEGORILER = {
    "sindirim": [
        "midem bulantı yaptı", "mide ağrısı", "kusma", "ishal oldum",
        "karın ağrısı", "mide yanması", "hazımsızlık", "iştahım kesildi",
        "şişkinlik", "kabız oldum", "mide krampı", "metalik tat",
        "yemek yiyemedim", "reflü", "gaz sorunu", "midemi yaktı"
    ],
    "norolojik": [
        "başım döndü", "baş ağrısı", "uyuyamadım", "halsizlik",
        "titreme", "sersemlik", "konsantre olamadım", "hafızam zayıfladı",
        "migren", "anksiyete", "depresyon", "konfüzyon", "baygınlık",
        "uykusuzluk", "kötü hissettim", "zihin bulanıklığı", "panik atak"
    ],
    "kardiyovaskuler": [
        "kalbim çarptı", "tansiyon yükseldi", "nefes alamadım",
        "nabız arttı", "kalp çarpıntısı", "tansiyonum düştü",
        "göğsüm sıkıştı", "ritim bozukluğu"
    ],
    "dermatolojik": [
        "döküntü çıktı", "kaşıntı oldu", "cildim kızardı",
        "saçım döküldü", "ürtiker çıktı", "ciltte soyulma",
        "alerjik reaksiyon", "şişme oldu", "derimde leke"
    ],
    "diger": [
        "yan etki yaşadım", "istenmeyen etki", "kilo aldım",
        "cinsel sorun", "ağız kuruluğu", "terleme arttı"
    ]
}

print("Kategori embeddinglari hesaplanıyor...")
kategori_embeddingler = {}
for kat, ornekler in KATEGORILER.items():
    embeddingler = model.encode(ornekler, convert_to_tensor=True)
    kategori_embeddingler[kat] = embeddingler
    print(f"  {kat}: {len(ornekler)} örnek")

with open("data/labeled/kategori_embeddingler.pkl", "wb") as f:
    pickle.dump(kategori_embeddingler, f)

print("\n[✓] data/labeled/kategori_embeddingler.pkl kaydedildi")
