import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt

# Define plot output directory
plot_output_dir = "/home/madhavimathur/Biobert_plots"
os.makedirs(plot_output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("/home/madhavimathur/Datasets/IPD_Brain.csv")

# Preprocessing function to convert structured data to text format
def structured_to_text(row):
    return (
        f"Patient {row['Case Number']} is a {row['Age']}-year-old "
        f"{'male' if str(row['Sex']).strip().upper() == 'M' else 'female'} who presented with {row['C/F']}. "
        f"Radiology findings: {row['Radiology'] if pd.notna(row['Radiology']) else 'not available'}. "
        f"Diagnosis confirms {row['Diagnosis']}. "
        f"Located at the {row['SITE']}, subtype: {row['Subtype']}. "
        f"Ki-67 proliferation index: {row['ki67(in %)']}%. "
        f"Molecular markers - IDH1 R132H: {row['IDH1R132H']}, ATRX: {row['ATRX']}, p53: {row['p53']}."
    )

# Drop rows with missing target columns
df = df.dropna(subset=["Diagnosis", "Subtype"])

# Convert structured data into text format
texts = df.apply(structured_to_text, axis=1).tolist()

# Encode labels
labels = df["Subtype"].astype("category").cat.codes.tolist()

# Create label2id and id2label mappings
label2id = {cat: idx for idx, cat in enumerate(df["Subtype"].astype("category").cat.categories)}
id2label = {v: k for k, v in label2id.items()}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# Custom dataset class
class BioTextDataset(Dataset):
    def __init__(self, texts, labels, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create datasets for training and validation
train_dataset = BioTextDataset(train_texts, train_labels)
val_dataset = BioTextDataset(val_texts, val_labels)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-v1.1",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="/home/madhavimathur/Biobertresults",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=120,
    logging_dir="/home/madhavimathur/Biobertlogs",
    logging_steps=10,
    save_total_limit=2
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Extract training and evaluation logs
logs = trainer.state.log_history

# Separate values for plotting
train_loss = [log["loss"] for log in logs if "loss" in log]
eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
eval_accuracy = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]

# Get steps for x-axis
train_steps = list(range(1, len(train_loss) + 1))
eval_steps = list(range(1, len(eval_loss) + 1))

# Plot training and evaluation loss
plt.figure(figsize=(10, 4))
plt.plot(train_steps, train_loss, label="Training Loss", color="blue", marker='o')
plt.plot(eval_steps, eval_loss, label="Validation Loss", color="red", marker='x')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(plot_output_dir, "loss_plot.png")
plt.savefig(loss_plot_path)
plt.close()

# Plot evaluation accuracy
if eval_accuracy:
    plt.figure(figsize=(6, 4))
    plt.plot(eval_steps, eval_accuracy, label="Validation Accuracy", color="green", marker='^')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    accuracy_plot_path = os.path.join(plot_output_dir, "accuracy_plot.png")
    plt.savefig(accuracy_plot_path)
    plt.close()