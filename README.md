# 💊 FarmaIQ — Türkçe Sosyal Medyada İlaç Yan Etki Madenciliği

> Türkçe ilaç yorumlarından yapay zeka ile yan etki tespiti ve prospektüs karşılaştırması

**İnönü Üniversitesi · Bilgisayar Mühendisliği · Yapay Zeka Dersi**

---

## 🎯 Proje Nedir?

Hastalar ilaç kullanım deneyimlerini sosyal medyada aktif olarak paylaşıyor. Bu yorumlar, klinik çalışmalarda gözden kaçabilen gerçek dünya yan etkilerini içeriyor. FarmaIQ, bu yorumları otomatik olarak analiz eden, yan etkileri tespit eden ve resmi TİTCK prospektüsleriyle karşılaştıran bir yapay zeka sistemidir.

**Problem:** Türkçe sosyal medyada paylaşılan ilaç yan etki raporları manuel olarak takip edilemiyor.

**Çözüm:** BERTurk tabanlı derin öğrenme modeli ile otomatik yan etki tespiti ve kategorizasyon.

---

## ✨ Özellikler

- 🔍 **Yan Etki Tespiti** — BERTurk fine-tune modeliyle %96 doğrulukla Türkçe yorumlardan yan etki sınıflandırması
- 🏷️ **Kategori Tespiti** — Semantik benzerlik ile sindirim, nörolojik, dermatolojik, kardiyovasküler kategorilerinde otomatik sınıflandırma
- 📋 **Prospektüs Karşılaştırması** — Tespit edilen yan etkinin resmi TİTCK prospektüsünde yer alıp almadığının tespiti
- 📊 **İnteraktif Dashboard** — React tabanlı arayüzde gerçek zamanlı analiz ve veri görselleştirme
- 🌐 **REST API** — FastAPI ile geliştirilmiş, kolayca entegre edilebilir backend

---

## 📊 Model Performansı

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| TF-IDF + Logistic Regression (Baseline) | 0.76 | 0.76 |
| **BERTurk Fine-Tune** | **0.96** | **0.96** |

BERTurk modeli baseline'a kıyasla **%20 doğruluk artışı** sağladı.

### Sınıf Bazında Performans (BERTurk)

| Sınıf | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Yan Etki Yok | 0.98 | 0.95 | 0.96 |
| Yan Etki Var | 0.93 | 0.98 | 0.95 |

---

## 🗂 Veri Seti

| Özellik | Değer |
|---------|-------|
| Toplam yorum | 476 |
| Yan etki var | ~200 |
| Yan etki yok | ~276 |
| Farklı ilaç sayısı | 25 |
| Dil | Türkçe |

### Veri Kaynakları

- **Sosyal Medya** — Türkçe forum ve sosyal medya platformlarından toplanan gerçek kullanıcı yorumları
- **Google Form** — Proje kapsamında doğrudan kullanıcılardan toplanan birincil veriler

### Desteklenen İlaçlar

`Amoksisilin` `Apranax` `Arveles` `Aspirin` `Augmentin` `Cipralex` `Cipram` `Diazepam` `İbuprofen` `Klaritromisin` `Lansoprazol` `Lustral` `Majezik` `Naproxen` `Nexium` `Nurofen` `Omeprazol` `Paracetamol` `Parol` `Paxil` `Prozac` `Ritalin` `Sertralin` `Voltaren` `Xanax`

### Yan Etki Kategorileri

| Kategori | Kapsam |
|----------|--------|
| 🟡 Sindirim | Bulantı, kusma, ishal, karın ağrısı, mide yanması |
| 🔵 Nörolojik | Baş dönmesi, baş ağrısı, uykusuzluk, titreme |
| 🔴 Kardiyovasküler | Çarpıntı, tansiyon değişimi, nefes darlığı |
| 🟣 Dermatolojik | Döküntü, kaşıntı, cilt reaksiyonları |
| ⚪ Diğer | Yukarıdaki kategorilere girmeyen yan etkiler |

---

## ⚙️ Kurulum

```bash
pip3 install -r requirements.txt

# Model eğitimi
python3 src/normalize.py
python3 src/kategori_model.py
python3 src/bertturk.py

# Backend
uvicorn src.api:app --port 8000 --host 0.0.0.0

# Frontend
cd ../frontend && npm start

# veya tek komutla
npm run dev
```

---

## 🔧 Teknolojiler

| Kategori | Teknoloji |
|----------|-----------|
| Ana Model | `dbmdz/bert-base-turkish-cased` (BERTurk) |
| Kategori Modeli | `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` |
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP | Sentence Transformers, Scikit-learn |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Recharts |
| Demo | Streamlit, Plotly |
| Veri İşleme | Pandas, NumPy |

---

## 📈 Geliştirme Süreci

1. **Veri Toplama** — Sosyal medya ve Google Form ile 476 Türkçe ilaç yorumu toplandı
2. **Veri Temizleme** — Gürültülü ve alakasız yorumlar filtrelendi
3. **Etiketleme** — Yan etki var/yok ve kategori etiketleri atandı
4. **Normalizasyon** — Kısa yorumlar cümle formatına dönüştürüldü
5. **Baseline** — TF-IDF + Logistic Regression ile %76 doğruluk elde edildi
6. **BERTurk Fine-Tune** — 4 epoch eğitimle %96 doğruluğa ulaşıldı
7. **Prospektüs Entegrasyonu** — 25 ilaç için TİTCK KÜB verileri işlendi
8. **API ve Arayüz** — FastAPI backend ve React frontend geliştirildi

---

## ⚠️ Sınırlılıklar

- Veri seti nispeten küçük (476 yorum), daha büyük veri ile performans artabilir
- Nadir yan etki kategorilerinde örnek sayısı az
- Olumsuzluk içeren bazı cümleler yanlış sınıflandırılabiliyor
- Prospektüs verileri 25 ilaçla sınırlı

---

## 👤 Geliştirici

**Mahmut Efe Türkol**
İnönü Üniversitesi · Bilgisayar Mühendisliği · 3. Sınıf

---

## 📄 Lisans

Bu proje akademik amaçlı geliştirilmiştir.
