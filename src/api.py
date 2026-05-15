from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pickle
import pandas as pd
import os
import os

app = FastAPI(title="FarmaIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROSPEKTUS = {
    "apranax": ["baş dönmesi", "baş ağrısı", "bulantı", "kusma", "mide yanması",
                "karın ağrısı", "diyare", "konstipasyon", "kaşıntı", "deri döküntüsü",
                "terleme", "saç dökülmesi", "ödem", "hipertansiyon", "çarpıntı",
                "uykusuzluk", "depresyon", "görme bulanıklığı", "kulak çınlaması"],
    "paxil": ["bulantı", "konstipasyon", "diyare", "kusma", "ağız kuruluğu",
              "baş ağrısı", "sersemlik", "titreme", "uykusuzluk", "terleme",
              "cinsel disfonksiyon", "kilo artışı", "iştah azalması", "deri döküntüsü",
              "bulanık görme", "çarpıntı"],
    "nexium": ["baş ağrısı", "ishal", "karın ağrısı", "kabızlık", "bulantı",
               "kusma", "baş dönmesi", "uykusuzluk", "kaşıntı", "deri döküntüsü",
               "artralji", "miyalji", "yorgunluk", "terleme", "saç dökülmesi"],
    "cipram": ["uyuşukluk", "uykusuzluk", "terleme", "ağız kuruluğu", "bulantı",
               "iştah azalması", "titreme", "baş dönmesi", "ishal", "kusma",
               "kabızlık", "kaşıntı", "yorgunluk", "saç dökülmesi", "halüsinasyon"],
    "prozac": ["bulantı", "ishal", "kusma", "ağız kuruluğu", "terleme",
               "titreme", "baş ağrısı", "sersemlik", "uyku sorunları",
               "iştah kaybı", "kilo kaybı", "halüsinasyon", "saç dökülmesi",
               "bulanık görme", "eklem ağrısı", "kas ağrısı"],
    "klaritromisin": ["ishal", "kusma", "karın ağrısı", "bulantı", "uykusuzluk",
                      "baş dönmesi", "kulak çınlaması", "titreme", "halsizlik",
                      "ağız kuruluğu", "tat alma bozukluğu", "duyma bozukluğu"],
    "augmentin": ["ishal", "bulantı", "kusma", "karın ağrısı", "deri döküntüsü",
                  "kaşıntı", "ürtiker", "baş ağrısı", "baş dönmesi", "ağız kuruluğu"],
    "majezik": ["bulantı", "baş dönmesi", "karın ağrısı", "diyare", "konstipasyon",
                "ürtiker", "baş ağrısı", "sinirlilik", "kulak çınlaması", "bulanık görme"],
    "parol": ["bulantı", "baş ağrısı", "karaciğer hasarı", "deri döküntüsü", "kaşıntı"],
    "xanax": ["uyuşukluk", "sersemlik", "baş dönmesi", "halsizlik", "bağımlılık",
              "hafıza sorunları", "konfüzyon", "baş ağrısı"],
    "sertralin": ["bulantı", "ishal", "baş dönmesi", "uykusuzluk", "terleme",
                  "cinsel disfonksiyon", "iştah azalması", "titreme", "ağız kuruluğu"],
    "omeprazol": ["baş ağrısı", "ishal", "bulantı", "kabızlık", "gaz",
                  "baş dönmesi", "deri döküntüsü", "kaşıntı"],
    "ibuprofen": ["bulantı", "karın ağrısı", "mide yanması", "baş dönmesi",
                  "baş ağrısı", "deri döküntüsü", "ödem", "hipertansiyon"],
    "naproxen": ["bulantı", "mide yanması", "karın ağrısı", "baş dönmesi",
                 "baş ağrısı", "deri döküntüsü", "ödem", "kulak çınlaması"],
    "amoksisilin": ["bulantı", "ishal", "deri döküntüsü", "kaşıntı", "ürtiker", "karın ağrısı"],
    "cipralex": ["bulantı", "uykusuzluk", "baş dönmesi", "baş ağrısı", "terleme",
                 "cinsel disfonksiyon", "iştah azalması", "yorgunluk", "titreme"],
    "lustral": ["bulantı", "ishal", "baş dönmesi", "uykusuzluk", "terleme",
                "cinsel disfonksiyon", "iştah azalması", "ağız kuruluğu"],
    "lansoprazol": ["baş ağrısı", "ishal", "bulantı", "kabızlık", "gaz", "baş dönmesi"],
    "diazepam": ["uyuşukluk", "sersemlik", "baş dönmesi", "halsizlik", "bağımlılık", "konfüzyon"],
    "ritalin": ["iştah kaybı", "uykusuzluk", "baş ağrısı", "sinirlilik", "çarpıntı", "tansiyon yükselmesi"],
    "paracetamol": ["bulantı", "karaciğer hasarı", "deri döküntüsü", "kaşıntı"],
    "voltaren": ["mide yanması", "bulantı", "baş dönmesi", "deri döküntüsü", "kaşıntı", "ödem"],
    "arveles": ["bulantı", "karın ağrısı", "baş dönmesi", "baş ağrısı", "mide yanması", "deri döküntüsü"],
}

KATEGORI_PROSPEKTUS = {
    "sindirim": ["bulantı", "kusma", "ishal", "karın ağrısı", "mide yanması",
                 "konstipasyon", "diyare", "iştah azalması", "ağız kuruluğu", "gaz"],
    "norolojik": ["baş dönmesi", "baş ağrısı", "uykusuzluk", "titreme",
                  "sersemlik", "halsizlik", "uyku sorunları", "konfüzyon"],
    "kardiyovaskuler": ["çarpıntı", "hipertansiyon", "ödem", "tansiyon yükselmesi"],
    "dermatolojik": ["deri döküntüsü", "kaşıntı", "saç dökülmesi", "terleme", "ürtiker"],
    "diger": []
}

print("Modeller yükleniyor...")
tokenizer = BertTokenizer.from_pretrained("data/labeled/bertturk_finetuned")
bert_model = BertForSequenceClassification.from_pretrained("data/labeled/bertturk_finetuned")
bert_model.eval()
st_model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
with open("data/labeled/kategori_embeddingler.pkl", "rb") as f:
    kategori_embeddingler = pickle.load(f)
print("Modeller hazır!")

class AnalizRequest(BaseModel):
    ilac: str
    yorum: str

def kategori_tahmin(metin):
    embedding = st_model.encode(metin, convert_to_tensor=True)
    en_iyi_kat = "diger"
    en_iyi_skor = 0.3
    for kat, embeddingler in kategori_embeddingler.items():
        benzerlik = util.cos_sim(embedding, embeddingler)
        max_skor = benzerlik.max().item()
        if max_skor > en_iyi_skor:
            en_iyi_skor = max_skor
            en_iyi_kat = kat
    return en_iyi_kat

def prospektus_karsilastir(kategori, ilac_lower):
    if ilac_lower not in PROSPEKTUS:
        return None, []
    ilac_etkiler = set(PROSPEKTUS[ilac_lower])
    kat_etkiler = set(KATEGORI_PROSPEKTUS.get(kategori, []))
    eslesme = list(ilac_etkiler & kat_etkiler)
    return bool(eslesme), eslesme

@app.get("/")
def root():
    return {"status": "FarmaIQ API çalışıyor"}

@app.post("/analyze")
def analyze(req: AnalizRequest):
    enc = tokenizer(req.yorum, max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = bert_model(**enc)
        probs = torch.softmax(out.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        guven = probs[pred].item()

    kategori = kategori_tahmin(req.yorum)
    ilac_lower = req.ilac.lower()
    prospektuste_var, eslesme = prospektus_karsilastir(kategori, ilac_lower)

    return {
        "yan_etki": bool(pred),
        "guven": round(guven * 100, 1),
        "kategori": kategori,
        "ilac": req.ilac,
        "prospektuste_var": prospektuste_var,
        "prospektus_eslesme": eslesme
    }

@app.get("/stats")
def stats():
    try:
        df = pd.read_csv("data/labeled/etiketlenmis.csv")
        ilac_dagilim = df[df["yan_etki"]==1].groupby("ilac").size().reset_index(name="sayi")
        kat_dagilim = df[df["yan_etki"]==1]["kategori"].value_counts().reset_index()
        return {
            "toplam": len(df),
            "yan_etki_var": int(df["yan_etki"].sum()),
            "yan_etki_yok": int((df["yan_etki"]==0).sum()),
            "ilac_sayisi": df["ilac"].nunique(),
            "ilac_dagilim": ilac_dagilim.to_dict(orient="records"),
            "kategori_dagilim": kat_dagilim.to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/ilaclar")
def ilaclar():
    return {"ilaclar": list(PROSPEKTUS.keys())}

from fastapi.responses import FileResponse

@app.get("/logo")
def get_logo():
    path = "prospektus/logo.png"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    from fastapi import HTTPException
    raise HTTPException(status_code=404)

from fastapi.responses import FileResponse

@app.get("/logo")
def get_logo():
    path = "prospektus/logo.png"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    from fastapi import HTTPException
    raise HTTPException(status_code=404)
