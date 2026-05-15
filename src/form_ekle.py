import pandas as pd

yeni_veri = """ilac,yorum,kaynak,yan_etki,kategori
prozac,Midem ağrıdı,google_form,1,sindirim
majezik,Uyku yaptı,google_form,1,norolojik
parol,Baş dönmesi,google_form,1,norolojik
nexium,Şişkinlik hissettim,google_form,1,sindirim
apranax,Mide bulantısı yaptı,google_form,1,sindirim
majezik,Mide bulantısı,google_form,1,sindirim
majezik,Mide bulantısı yaptı,google_form,1,sindirim
nurofen,midem bulandı,google_form,1,sindirim
paxil,Kilo aldırttı,google_form,1,diger
augmentin,Mide bulantısı,google_form,1,sindirim
nexium,Kabızlık,google_form,1,sindirim
aspirin,Çarpıntı nabızda düzensizlik,google_form,1,kardiyovaskuler
prozac,Deri döküntüsü yaptı,google_form,1,dermatolojik
prozac,Çarpıntı yaptı,google_form,1,kardiyovaskuler
prozac,Kaşıntı oldu,google_form,1,dermatolojik
prozac,baş ağrısı ve uykusuzluk,google_form,1,norolojik
augmentin,Midem bulandı kalp çarpıntısı oldu,google_form,1,sindirim
cipram,Bulantı yaptı,google_form,1,sindirim
cipram,Uyku bozukluğu yaptı,google_form,1,norolojik
parol,Uyku bozukluğu,google_form,1,norolojik
augmentin,Kaşıntı yaptı,google_form,1,dermatolojik
augmentin,Midede yanma yaptı,google_form,1,sindirim
augmentin,İshal,google_form,1,sindirim
augmentin,Deri döküntüsü,google_form,1,dermatolojik
augmentin,Mide bulantısı kusma yaptı,google_form,1,sindirim
augmentin,Cildimde kızarıklık oldu,google_form,1,dermatolojik
nurofen,Mide bulantısı,google_form,1,sindirim
prozac,Bişey olmadı,google_form,0,yok
nurofen,Hazımsızlık,google_form,1,sindirim
nurofen,Baş ağrısı,google_form,1,norolojik
nurofen,Baş dönmesi,google_form,1,norolojik
aspirin,Midede yanma ve ağrı yaptı,google_form,1,sindirim
parol,Tansiyonum düştü,google_form,1,kardiyovaskuler
majezik,Baş dönmesi yaptı,google_form,1,norolojik
aspirin,Diş eti kanaması yaptı,google_form,1,dermatolojik
aspirin,Ciltte döküntü,google_form,1,dermatolojik
aspirin,Ciltte ödem,google_form,1,dermatolojik
majezik,Mide bulantısı ağrısı,google_form,1,sindirim
nurofen,Çarpıntı,google_form,1,kardiyovaskuler
augmentin,Mide bulantısı,google_form,1,sindirim
augmentin,Başım döndü,google_form,1,norolojik
apranax,Mide ağrısı yaptı,google_form,1,sindirim
aspirin,Derimde soyulma ve kızarıklıklar meydana geldi,google_form,1,dermatolojik
nurofen,Kalp çarpıntısı,google_form,1,kardiyovaskuler
paxil,Mide bulantısı ve baş dönmesi yaşadım,google_form,1,sindirim
majezik,Mide bulantısı yaptı,google_form,1,sindirim
majezik,Hafif bulantı,google_form,0,yok
parol,ishal oldum,google_form,1,sindirim
parol,Hayır,google_form,0,yok
nurofen,Carpıntı,google_form,1,kardiyovaskuler
majezik,kabız oldum,google_form,1,sindirim
nurofen,Midem bulandı,google_form,1,sindirim
cipram,Mide bulantısı kalp çarpıntısı,google_form,1,sindirim
parol,Başım ağrıyordu kullandıktan 25 dakika sonra bir ağrım kalmadı,google_form,0,yok
prozac,Mide bulantısı yaptı,google_form,1,sindirim
parol,bulantı yaptı,google_form,1,sindirim
arveles,Başım döndü,google_form,1,norolojik
arveles,Nabız artışı yaşadım,google_form,1,kardiyovaskuler
nexium,mide bulantısı,google_form,1,sindirim
arveles,Mide bulantısı yaptı,google_form,1,sindirim
augmentin,Mide bulantısı,google_form,1,sindirim
nurofen,başım döndü,google_form,1,norolojik
parol,hiçbir etkisi olmadı,google_form,0,yok
augmentin,"kullandıktan sonra midem çok bulantı yaptı, yemek yiyemez hale geldim ve sürekli karın ağrısı çektim",google_form,1,sindirim
nexium,"içmeye başladıktan birkaç gün sonra şiddetli ishal oldum, mide krampları had safhaya ulaştı",google_form,1,sindirim
prozac,"kullanırken ağzımda metalik bir tat oluştu, iştahım tamamen kesildi ve kusma isteği geçmedi",google_form,1,sindirim
parol,Midem bulandı,google_form,1,sindirim
paxil,"kullanmaya başladıktan sonra sürekli baş dönmesi yaşadım, gece uyuyamıyordum ve sabah kalkmak çok zorlaştı",google_form,1,norolojik
cipram,"içerken konsantre olmakta büyük güçlük çektim, hafızam zayıfladı ve sürekli sersemlik hissettim",google_form,1,norolojik
sertralin,İlacı içtikten sonra aşırı şekilde hareket etme isteği ve uykusuzluk geldi,google_form,1,norolojik
nurofen,"kullandığım dönemde aşırı halsizlik yaşadım, günde 12 saat uyumama rağmen dinçlik hissedemedim",google_form,1,norolojik
apranax,"kullandıktan sonra kollarımda kızarıklık ve kaşıntı başladı, zamanla döküntüye dönüştü",google_form,1,dermatolojik
aspirin,"kullanırken tüm vücudumda ürtiker çıktı, cildin bazı bölgeleri şişti ve yanma hissi oluştu",google_form,1,dermatolojik
majezik,"aldıktan sonra kalbim çarpıntı yapmaya başladı, tansiyonum yükseldi ve nefes almak güçleşti",google_form,1,kardiyovaskuler
cipram,"aldıktan sonra kalbim çarpıntı yapmaya başladı, tansiyonum yükseldi ve nefes almak güçleşti",google_form,1,kardiyovaskuler
apranax,"ağrımı çok hızlı geçirdi, hiçbir yan etki yaşamadım gayet memnun kaldım",google_form,0,yok
nexium,"mide yanmamı tamamen geçirdi, kullandığım süre boyunca herhangi bir şikayet olmadı",google_form,0,yok
aspirin,"mide yanmamı tamamen geçirdi, kullandığım süre boyunca herhangi bir şikayet olmadı",google_form,0,yok
nurofen,Karnım guruldadı,google_form,1,sindirim
nurofen,Bir şey olmadı,google_form,0,yok
xanax,"kullandıktan sonra midem çok bulantı yaptı, yemek yiyemez hale geldim ve sürekli karın ağrısı çektim",google_form,1,sindirim
xanax,"içmeye başladıktan birkaç gün sonra şiddetli ishal oldum, mide krampları had safhaya ulaştı",google_form,1,sindirim
prozac,Mide bulantısı uykusuzluk,google_form,1,norolojik
paxil,Diş ağrısı ve döküntü yaptı,google_form,1,dermatolojik
parol,Yan etki olmadı,google_form,0,yok
sertralin,migrenimi tetikledi,google_form,1,norolojik
concerta,nabız artışı ve kalp çarpıntısı yaşadım,google_form,1,kardiyovaskuler"""

import io
yeni_df = pd.read_csv(io.StringIO(yeni_veri))

# Gereksiz/anlamsız yorumları temizle
temizle = ["Evet", "Hayır", "Efetürkol kaan selamlar", "çırçır oldum",
           "enfeksiyona sebep oldu", "Ateş düşüklüğü yaşadım", "ibucold",
           "concerta", "aspirin"]
yeni_df = yeni_df[~yeni_df["yorum"].isin(["Evet", "Hayır", "Efetürkol kaan selamlar", "çırçır oldum"])]
yeni_df = yeni_df[~yeni_df["ilac"].isin(["ibucold", "concerta"])]

# Mevcut veri setiyle birleştir
mevcut = pd.read_csv("data/labeled/etiketlenmis.csv")
birlesik = pd.concat([mevcut, yeni_df], ignore_index=True)
birlesik = birlesik.drop_duplicates(subset=["yorum"])

print(f"Mevcut: {len(mevcut)} satır")
print(f"Yeni eklenen: {len(yeni_df)} satır")
print(f"Toplam (duplicate temizlendi): {len(birlesik)} satır")
print(f"\nYan etki var (1): {birlesik['yan_etki'].sum()}")
print(f"Yan etki yok (0): {(birlesik['yan_etki']==0).sum()}")
print(f"\nKategori dağılımı:\n{birlesik['kategori'].value_counts()}")

birlesik.to_csv("data/labeled/etiketlenmis.csv", index=False, encoding="utf-8-sig")
print("\n[✓] data/labeled/etiketlenmis.csv güncellendi")
