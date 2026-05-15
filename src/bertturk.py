import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

df = pd.read_csv("data/labeled/etiketlenmis.csv")
df = df.dropna(subset=["yan_etki"])
df["yan_etki"] = df["yan_etki"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df["yorum"].tolist(), df["yan_etki"].tolist(),
    test_size=0.2, random_state=42, stratify=df["yan_etki"]
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

class IlacDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=128,
                        padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(IlacDataset(X_train, y_train), batch_size=8, shuffle=True)
test_loader  = DataLoader(IlacDataset(X_test, y_test), batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = BertForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

epoch_losses = []

for epoch in range(4):
    model.train()
    total_loss = 0
    for batch in train_loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["label"].to(device)
        optimizer.zero_grad()
        out  = model(ids, attention_mask=mask, labels=labs)
        out.loss.backward()
        optimizer.step()
        total_loss += out.loss.item()
    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/4 - Loss: {avg_loss:.4f}")

# ─── EĞİTİM LOSS GRAFİĞİ ───
plt.figure(figsize=(8, 4))
plt.plot(range(1, 5), epoch_losses, marker='o', color='#2563eb', linewidth=2, markersize=8)
plt.fill_between(range(1, 5), epoch_losses, alpha=0.1, color='#2563eb')
plt.title("BERTurk Eğitim Loss Grafiği", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.xticks(range(1, 5))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/labeled/egitim_loss.png", dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Eğitim loss grafiği kaydedildi")

model.eval()
preds, labels = [], []
with torch.no_grad():
    for batch in test_loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out  = model(ids, attention_mask=mask)
        preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        labels.extend(batch["label"].numpy())

print("\n=== BERTurk Fine-Tune Sonuçları ===")
print(classification_report(labels, preds,
      target_names=["Yan Etki Yok", "Yan Etki Var"]))

# ─── CONFUSION MATRIX ───
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Yan Etki Yok", "Yan Etki Var"],
            yticklabels=["Yan Etki Yok", "Yan Etki Var"],
            annot_kws={"size": 14, "weight": "bold"})
plt.title("Confusion Matrix — BERTurk", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Gerçek", fontsize=12)
plt.xlabel("Tahmin", fontsize=12)
plt.tight_layout()
plt.savefig("data/labeled/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Confusion matrix kaydedildi")

model.save_pretrained("data/labeled/bertturk_finetuned")
tokenizer.save_pretrained("data/labeled/bertturk_finetuned")
print("[✓] Model kaydedildi → data/labeled/bertturk_finetuned/")
