import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/labeled/etiketlenmis.csv")
df = df.dropna(subset=["yan_etki"])
df["yan_etki"] = df["yan_etki"].astype(int)

X = df["yorum"]
y = df["yan_etki"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("=== BASELINE: TF-IDF + Lojistik Regresyon ===\n")
print(classification_report(y_test, y_pred, target_names=["Yan Etki Yok", "Yan Etki Var"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Yok", "Var"], yticklabels=["Yok", "Var"])
plt.title("Baseline Confusion Matrix")
plt.ylabel("Gerçek")
plt.xlabel("Tahmin")
plt.tight_layout()
plt.savefig("data/labeled/baseline_confusion.png")
print("\n[✓] Confusion matrix → data/labeled/baseline_confusion.png")
