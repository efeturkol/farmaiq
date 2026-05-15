import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
    print(f"Epoch {epoch+1}/4 - Loss: {total_loss/len(train_loader):.4f}")

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

# Doğru kaydetme — tüm modeli kaydet
model.save_pretrained("data/labeled/bertturk_finetuned")
tokenizer.save_pretrained("data/labeled/bertturk_finetuned")
print("[✓] Model kaydedildi → data/labeled/bertturk_finetuned/")
