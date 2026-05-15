import streamlit as st
import pandas as pd
import pickle
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go

st.set_page_config(page_title="FarmaIQ", page_icon="💊", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');section[data-testid="stMain"] > div:first-child { padding-top: 0 !important; }
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #f8f9fb !important;
    color: #1a1d23 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { background: #f8f9fb !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; }
.fiq-header { display:flex; align-items:center; gap:13px; margin-bottom:1.2rem; padding-bottom:1rem; border-bottom:1px solid #e8eaef; }
.fiq-icon { width:42px; height:42px; background:#2563eb; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; }
.fiq-title { font-size:1.4rem; font-weight:700; color:#0f172a; margin:0; letter-spacing:-0.02em; }
.fiq-sub { font-size:0.76rem; color:#94a3b8; margin-top:2px; }
.sec-title { font-size:0.68rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem; }
[data-testid="stSelectbox"] > div > div {
    background:#fff !important; border:1px solid #e2e6ed !important;
    border-radius:8px !important; color:#1a1d23 !important; font-family:'Inter',sans-serif !important; font-size:0.88rem !important;
}
[data-testid="stSelectbox"] > div > div:hover { border-color:#2563eb !important; }
[data-testid="stTextArea"] textarea {
    background:#fff !important; border:1px solid #e2e6ed !important;
    border-radius:8px !important; color:#1a1d23 !important; font-family:'Inter',sans-serif !important;
    font-size:0.88rem !important; line-height:1.6 !important; resize:vertical !important;
}
[data-testid="stTextArea"] textarea:focus { border-color:#2563eb !important; box-shadow:0 0 0 3px rgba(37,99,235,0.08) !important; }
[data-testid="stTextArea"] textarea::placeholder { color:#c8cfd8 !important; }
[data-testid="stButton"] > button {
    background:#2563eb !important; color:#fff !important; border:none !important;
    border-radius:8px !important; font-family:'Inter',sans-serif !important; font-weight:600 !important;
    font-size:0.86rem !important; padding:0.6rem 1.5rem !important; width:100% !important;
}
[data-testid="stButton"] > button:hover { background:#1d4ed8 !important; }
[data-testid="stWidgetLabel"] p {
    font-family:'Inter',sans-serif !important; font-size:0.76rem !important;
    font-weight:600 !important; color:#64748b !important; text-transform:uppercase !important; letter-spacing:0.07em !important;
}
.res-box { border-radius:10px; padding:1.2rem 1.4rem; margin-top:0.2rem; }
.res-positive { background:#fef2f2; border:1px solid #fecaca; border-left:3px solid #ef4444; }
.res-negative { background:#f0fdf4; border:1px solid #bbf7d0; border-left:3px solid #22c55e; }
.res-title { font-size:0.95rem; font-weight:700; color:#0f172a; margin-bottom:3px; }
.res-sub { font-size:0.78rem; color:#64748b; }
.conf-row { display:flex; justify-content:space-between; font-size:0.73rem; color:#94a3b8; margin:0.8rem 0 4px; }
.conf-bg { background:#e8ecf2; border-radius:999px; height:5px; overflow:hidden; }
.conf-red { height:100%; border-radius:999px; background:#ef4444; }
.conf-green { height:100%; border-radius:999px; background:#22c55e; }
.detail-row { display:flex; gap:0.8rem; margin-top:0.8rem; }
.detail-card { flex:1; background:#fff; border:1px solid #e8eaef; border-radius:9px; padding:0.9rem 1rem; }
.detail-lbl { font-size:0.68rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px; }
.b-blue { display:inline-block; background:#eff6ff; border:1px solid #bfdbfe; color:#2563eb; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
.b-green { display:inline-block; background:#f0fdf4; border:1px solid #bbf7d0; color:#16a34a; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
.b-yellow { display:inline-block; background:#fffbeb; border:1px solid #fde68a; color:#d97706; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
.b-gray { display:inline-block; background:#f8fafc; border:1px solid #e2e8f0; color:#94a3b8; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
.metric-card { background:#fff; border:1px solid #e8eaef; border-radius:10px; padding:1rem; text-align:center; }
.metric-val { font-size:1.7rem; font-weight:700; color:#0f172a; line-height:1; }
.metric-lbl { font-size:0.7rem; color:#94a3b8; margin-top:4px; text-transform:uppercase; letter-spacing:0.07em; }
.info-card { background:#fff; border:1px solid #e8eaef; border-radius:11px; padding:1.2rem 1.3rem; margin-bottom:0.8rem; }
.info-card h4 { font-size:0.72rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.08em; margin:0 0 0.8rem 0; }
.info-card li, .info-card p { font-size:0.82rem; color:#64748b; line-height:1.7; margin:0; }
.info-card ul { padding-left:1.2rem; margin:0; }
.info-card li { margin-bottom:0.25rem; }
.perf-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.perf-table th { font-size:0.7rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:0.07em; padding:0.4rem 0.7rem; border-bottom:1px solid #e8eaef; text-align:left; }
.perf-table td { padding:0.5rem 0.7rem; color:#475569; border-bottom:1px solid #f1f5f9; }
.perf-table td:first-child { color:#1e293b; font-weight:500; }
.perf-good { color:#16a34a !important; font-weight:700 !important; }
.divider { border:none; border-top:1px solid #e8eaef; margin:1.2rem 0; }
.fiq-footer { text-align:center; padding:1.5rem 0 0.5rem; border-top:1px solid #e8eaef; margin-top:2rem; font-size:0.73rem; color:#cbd5e1; }
@media (max-width:768px) { .block-container { padding:1.2rem !important; } }
</style>
""", unsafe_allow_html=True)

ILACLAR = ["Apranax","Majezik","Parol","Voltaren","Arveles","Cipralex","Prozac","Sertralin","Paxil","Lustral","Cipram","Augmentin","Amoksisilin","Klaritromisin","Nexium","Lansoprazol","Omeprazol","Xanax","Diazepam","Ritalin","Paracetamol","İbuprofen","Naproxen"]

PROSPEKTUS = {
    "Apranax":{"sindirim","kardiyovaskuler","norolojik"},"Majezik":{"sindirim","kardiyovaskuler","dermatolojik"},
    "Parol":{"sindirim","norolojik"},"Voltaren":{"sindirim","kardiyovaskuler","dermatolojik"},
    "Arveles":{"sindirim","kardiyovaskuler"},"Cipralex":{"norolojik","sindirim","diger"},
    "Prozac":{"norolojik","sindirim","diger"},"Sertralin":{"norolojik","sindirim"},
    "Paxil":{"norolojik","sindirim","diger"},"Lustral":{"norolojik","sindirim"},
    "Cipram":{"norolojik","sindirim"},"Augmentin":{"sindirim","dermatolojik"},
    "Amoksisilin":{"sindirim","dermatolojik"},"Klaritromisin":{"sindirim","norolojik","dermatolojik"},
    "Nexium":{"sindirim"},"Lansoprazol":{"sindirim","norolojik"},
    "Omeprazol":{"sindirim","norolojik"},"Xanax":{"norolojik","diger"},
    "Diazepam":{"norolojik","diger"},"Ritalin":{"norolojik","kardiyovaskuler","diger"},
    "Paracetamol":{"sindirim","dermatolojik"},"İbuprofen":{"sindirim","kardiyovaskuler","dermatolojik"},
    "Naproxen":{"sindirim","kardiyovaskuler"},
}

KAT_TR = {"sindirim":"Sindirim","norolojik":"Nörolojik","dermatolojik":"Dermatolojik","kardiyovaskuler":"Kardiyovasküler","diger":"Diğer","yok":"Yok"}

@st.cache_resource(show_spinner=False)
def load_bert_model():
    tok = BertTokenizer.from_pretrained("data/labeled/bertturk_finetuned")
    mdl = BertForSequenceClassification.from_pretrained("data/labeled/bertturk_finetuned")
    mdl.eval()
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_sentence_model():
    return SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

@st.cache_resource(show_spinner=False)
def load_embeddings():
    with open("data/labeled/kategori_embeddingler.pkl","rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv("data/labeled/etiketlenmis.csv")

def bert_predict(tok, mdl, text):
    inp = tok(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        out = mdl(**inp)
    probs = torch.softmax(out.logits, dim=1).squeeze().numpy()
    label = int(np.argmax(probs))
    return label, float(probs[label])

def kategori_tespit(smdl, embeddings, text):
    emb = smdl.encode(text, convert_to_tensor=True)
    best_kat, best_score = "yok", -1.0
    for kat, e in embeddings.items():
        if kat == "yok": continue
        s = util.cos_sim(emb, e)[0][0].item()
        if s > best_score:
            best_score = s
            best_kat = kat
    if best_score < 0.3:
        best_kat = "yok"
    return best_kat, best_score

logo_path = "prospektus/logo.png"
if os.path.exists(logo_path):
    import base64
    with open(logo_path,"rb") as f:
        lb64 = base64.b64encode(f.read()).decode()
    icon_html = f'<img src="data:image/png;base64,{lb64}" style="width:42px;height:42px;border-radius:10px;object-fit:cover;">'
else:
    icon_html = '<div class="fiq-icon">💊</div>'

st.markdown(f'<div class="fiq-header">{icon_html}<div><div class="fiq-title">FarmaIQ</div><div class="fiq-sub">Türkçe Sosyal Medya · Yan Etki Tespit Sistemi</div></div></div>', unsafe_allow_html=True)

with st.spinner("Modeller yükleniyor…"):
    try:
        tokenizer, bert_model = load_bert_model()
        sent_model = load_sentence_model()
        kat_emb = load_embeddings()
        df = load_dataset()
        ok = True
    except Exception as e:
        ok = False; err = str(e)

if not ok:
    st.error(f"Model yüklenemedi: {err}")
    st.stop()

col_left, col_right = st.columns([1,1], gap="large")

with col_left:
    st.markdown('<div class="sec-title">Analiz</div>', unsafe_allow_html=True)
    secili_ilac = st.selectbox("İlaç", ILACLAR)
    yorum = st.text_area("Hasta Yorumu", placeholder="Yorumu buraya girin… (örn: Bu ilacı kullandıktan sonra midem fena oldu, bulantı yaşadım.)", height=115)
    btn = st.button("Analiz Et")

    if btn:
        if not yorum.strip():
            st.warning("Lütfen bir yorum girin.")
        else:
            with st.spinner("Analiz ediliyor…"):
                label, conf = bert_predict(tokenizer, bert_model, yorum)
                kat, kat_skor = kategori_tespit(sent_model, kat_emb, yorum)
                if label == 0: kat = "yok"

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Sonuç</div>', unsafe_allow_html=True)

            if label == 1:
                st.markdown(f'<div class="res-box res-positive"><div class="res-title">⚠️ Yan Etki Tespit Edildi</div><div class="res-sub">BERTurk modeli bu yorumda yan etki işareti buldu</div><div class="conf-row"><span>Güven Skoru</span><span>{conf*100:.1f}%</span></div><div class="conf-bg"><div class="conf-red" style="width:{conf*100:.1f}%"></div></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="res-box res-negative"><div class="res-title">✅ Yan Etki Tespit Edilmedi</div><div class="res-sub">Bu yorumda yan etki işareti bulunmadı</div><div class="conf-row"><span>Güven Skoru</span><span>{conf*100:.1f}%</span></div><div class="conf-bg"><div class="conf-green" style="width:{conf*100:.1f}%"></div></div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            dc1, dc2 = st.columns(2)
            kat_display = KAT_TR.get(kat, kat)

            with dc1:
                sim_html = f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:5px;">Benzerlik: {kat_skor:.2f}</div>' if label==1 and kat!="yok" else ""
                st.markdown(f'<div class="detail-card"><div class="detail-lbl">Kategori</div><span class="b-blue">{kat_display}</span>{sim_html}</div>', unsafe_allow_html=True)

            with dc2:
                if label == 0 or kat == "yok":
                    pros_html = '<span class="b-gray">Yan etki yok</span>'
                    pros_note = ""
                elif kat in PROSPEKTUS.get(secili_ilac, set()):
                    pros_html = '<span class="b-green">✓ Prospektüste mevcut</span>'
                    pros_note = f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:5px;">{secili_ilac} için bilinen yan etki.</div>'
                else:
                    pros_html = '<span class="b-yellow">⚡ Prospektüste yok</span>'
                    pros_note = f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:5px;">{secili_ilac} için beklenmedik yan etki.</div>'
                st.markdown(f'<div class="detail-card"><div class="detail-lbl">Prospektüs</div>{pros_html}{pros_note if label==1 and kat!="yok" else ""}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Veri Seti</div>', unsafe_allow_html=True)

    toplam = len(df)
    yan_var = int(df["yan_etki"].sum()) if "yan_etki" in df.columns else 0
    yan_yok = toplam - yan_var

    m1, m2, m3 = st.columns(3)
    for col, val, lbl in zip([m1,m2,m3],[toplam,yan_var,yan_yok],["Toplam","Yan Etki Var","Yan Etki Yok"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    if "ilac" in df.columns and "yan_etki" in df.columns:
        dfg = df.groupby("ilac")["yan_etki"].agg(["sum","count"]).reset_index()
        dfg.columns = ["ilac","yv","tot"]
        dfg = dfg.sort_values("tot", ascending=True).tail(12)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=dfg["ilac"], x=dfg["tot"]-dfg["yv"], name="Yan Etki Yok", orientation="h", marker_color="#bfdbfe"))
        fig.add_trace(go.Bar(y=dfg["ilac"], x=dfg["yv"], name="Yan Etki Var", orientation="h", marker_color="#fca5a5"))
        fig.update_layout(
            barmode="stack", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#64748b", size=11), margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(size=10),bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#f1f5f9",zeroline=False), yaxis=dict(gridcolor="rgba(0,0,0,0)"), height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown('<div class="sec-title">Nasıl Çalışır?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>Sistem Akışı</h4>
        <ul>
            <li><b>Adım 1 — Sınıflandırma:</b> BERTurk fine-tune modeli yorumu alır, yan etki var/yok tahmini üretir.</li>
            <li><b>Adım 2 — Kategori Tespiti:</b> Sentence Transformer ile semantic benzerlik hesaplanır.</li>
            <li><b>Adım 3 — Prospektüs Karşılaştırması:</b> Tespit edilen kategori ilaç prospektüsündeki bilinen yan etkilerle eşleştirilir.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Model Performansı</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>Karşılaştırmalı Sonuçlar</h4>
        <table class="perf-table">
            <tr><th>Model</th><th>Accuracy</th><th>Macro F1</th></tr>
            <tr><td>TF-IDF + Logistic Regression</td><td>0.76</td><td>0.76</td></tr>
            <tr><td>BERTurk Fine-Tune</td><td class="perf-good">0.96</td><td class="perf-good">0.96</td></tr>
        </table>
        <p style="margin-top:0.8rem;">Baseline'a kıyasla <b style="color:#16a34a;">+%20</b> doğruluk artışı.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Desteklenen Kategoriler</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>Yan Etki Sınıfları</h4>
        <div style="display:flex;flex-wrap:wrap;gap:6px;">
            <span class="b-blue">Sindirim</span>
            <span class="b-blue">Nörolojik</span>
            <span class="b-blue">Dermatolojik</span>
            <span class="b-blue">Kardiyovasküler</span>
            <span class="b-blue">Diğer</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "kategori" in df.columns:
        kd = df[df["yan_etki"]==1]["kategori"].value_counts().reset_index()
        kd.columns = ["kategori","sayi"]
        kd["kategori"] = kd["kategori"].map(KAT_TR).fillna(kd["kategori"])
        colors = ["#3b82f6","#60a5fa","#93c5fd","#bfdbfe","#dbeafe"]
        fig2 = go.Figure(go.Pie(
            labels=kd["kategori"], values=kd["sayi"], hole=0.5,
            marker=dict(colors=colors[:len(kd)], line=dict(color="#f8f9fb",width=2)),
            textfont=dict(family="Inter",size=11),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(font=dict(family="Inter",size=10,color="#64748b"),bgcolor="rgba(0,0,0,0)"),
            annotations=[dict(text="Dağılım",x=0.5,y=0.5,font=dict(family="Inter",size=12,color="#94a3b8"),showarrow=False)],
            height=210,
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="fiq-footer">İnönü Üniversitesi · Bilgisayar Mühendisliği · Yapay Zeka Dersi · 2025–2026 &nbsp;|&nbsp; FarmaIQ</div>', unsafe_allow_html=True)
