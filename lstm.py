import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Plot output directory
plot_output_dir = "/home/madhavimathur/LSTM_plots"
os.makedirs(plot_output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("/home/madhavimathur/Datasets/IPD_Brain.csv")
df = df.dropna(subset=["Diagnosis", "WHO Grade"])

# Convert structured data to natural text
def structured_to_text(row):
    return (
        f"Patient {row['Case Number']} is a {row['Age']}-year-old "
        f"{'male' if str(row['Sex']).strip().upper() == 'M' else 'female'} "
        f"presenting with {row['C/F']}. Radiology: "
        f"{row['Radiology'] if pd.notna(row['Radiology']) else 'not available'}. "
        f"Diagnosis: {row['Diagnosis']}. Site: {row['SITE']}, "
        f"Subtype: {row['Subtype']}. Ki-67: {row['ki67(in %)']}%. "
        f"Markers—IDH1 R132H: {row['IDH1R132H']}, "
        f"ATRX: {row['ATRX']}, p53: {row['p53']}."
    )

texts = df.apply(structured_to_text, axis=1).tolist()
labels = df["WHO Grade"].astype("category").cat.codes.tolist()
n_classes = len(set(labels))

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Dataset class
class LSTMDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.examples = [
            (
                torch.tensor(
                    tokenizer(t, truncation=True, padding=False, add_special_tokens=True)["input_ids"],
                    dtype=torch.long
                ),
                torch.tensor(l, dtype=torch.long)
            )
            for t, l in zip(texts, labels)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# Collate function for padding
def collate_fn(batch):
    input_ids, labels = zip(*batch)
    padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.stack(labels)
    return padded, labels

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

train_ds = LSTMDataset(train_texts, train_labels, tokenizer)
val_ds = LSTMDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)

# LSTM classifier
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        lstm_out, _ = self.lstm(x)
        feat = lstm_out[:, -1, :]
        return self.classifier(feat)

# Model setup
model = LSTMClassifier(vocab_size=len(tokenizer), num_classes=n_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

train_losses = []
val_accuracies = []

# Training loop
epochs = 500
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for input_ids, labels in train_loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids)
        loss = lossf(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch}/{epochs}  train_loss: {avg_train_loss:.4f}  val_acc: {val_acc:.4f}")

# Plotting results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy", color="green")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Over Epochs")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, "training_validation_plots.png"))
plt.close()
