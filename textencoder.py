import pandas as pd
import os
import pandas as pd
import os
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re

df = pd.read_csv("/home/madhavimathur/Datasets/IPD_Brain.csv")

def structured_to_text(row):
    return (
        f"Patient {row['Case Number']} is a {row['Age']}-year-old "
        f"{'male' if str(row['Sex']).strip().upper() == 'M' else 'female'} who presented with {row['C/F']}. "
        f"Radiology findings: {row['Radiology'] if pd.notna(row['Radiology']) else 'not available'}. "
        f"Diagnosis confirms {row['Diagnosis']}. "
        f"WHO grade: {row['WHO Grade']}, located at the {row['SITE']}, subtype: {row['Subtype']}. "
        f"Ki-67 proliferation index: {row['ki67(in %)']}%. "
        f"Molecular markers - IDH1 R132H: {row['IDH1R132H']}, ATRX: {row['ATRX']}, p53: {row['p53']}."
    )

for _, row in df.iterrows():
    text = structured_to_text(row)
    print(text)
    print()

image_dir = "/home/madhavimathur/Datasets/labelled"

image_to_text = {}
for _, row in df.iterrows():
    case_numbers = str(row['Case Number']).strip().replace('"', '').split("\n")
    text = structured_to_text(row)

    for case_number in case_numbers:
        case_number = case_number.strip()
        image_path = os.path.join(image_dir, f"{case_number}.png")

        if os.path.exists(image_path):
            image_to_text[image_path] = text
            print(f"{image_path}:\n{text}\n")

        else:
            print(f"[ERROR] Image not found for case: {case_number} → {image_path}")

for path, description in image_to_text.items():
    print(f"{path}:\n{description}\n")
    

def simple_tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

all_tokens = set()
for text in image_to_text.values():
    all_tokens.update(simple_tokenizer(text))

word2idx = {word: idx + 1 for idx, word in enumerate(sorted(all_tokens))}
word2idx['<PAD>'] = 0
vocab_size = len(word2idx)

def encode(text, max_len=100):
    tokens = simple_tokenizer(text)
    idxs = [word2idx.get(token, 0) for token in tokens[:max_len]]
    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))
    return torch.tensor(idxs)

class ClinicalTextDataset(Dataset):
    def __init__(self, data_dict):
        self.paths = list(data_dict.keys())
        self.texts = [encode(txt) for txt in data_dict.values()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx], self.texts[idx]

dataset = ClinicalTextDataset(image_to_text)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return hidden.squeeze(0)

from PIL import UnidentifiedImageError

model = TextEncoder(vocab_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

embedding_dict = {}

model.eval()
with torch.no_grad():
    for paths, encoded_texts in loader:
        encoded_texts = encoded_texts.to(device)
        embeddings = model(encoded_texts)
        for path, embed in zip(paths, embeddings):
            try:
                with Image.open(path) as img:
                    img.verify()
                embedding_dict[path] = embed.cpu()
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                print(f"[ERROR] Cannot open image at: {path} — {str(e)}")

for path, vector in embedding_dict.items():
    print(path)
    print(vector[:10])
    print()
    
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import torch

# Create output directory if it doesn't exist
output_dir = "textencoder_outputs"
os.makedirs(output_dir, exist_ok=True)

# t-SNE
vectors = torch.stack(list(embedding_dict.values()))
tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vectors)

plt.figure()
plt.scatter(reduced[:, 0], reduced[:, 1])
plt.title("Text Embedding Visualization (t-SNE)")
plt.savefig(os.path.join(output_dir, "tsne_plot.png"))
plt.close()

# PCA (2D)
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

plt.figure()
plt.scatter(reduced[:, 0], reduced[:, 1])
plt.title("PCA of Clinical Text Embeddings")
plt.savefig(os.path.join(output_dir, "pca_2d_plot.png"))
plt.close()

# UMAP
import umap

reducer = umap.UMAP()
reduced = reducer.fit_transform(vectors)


plt.figure()
plt.scatter(reduced[:, 0], reduced[:, 1])
plt.title("UMAP of Clinical Text Embeddings")
plt.savefig(os.path.join(output_dir, "umap_plot.png"))
plt.close()

# PCA (3D)
pca = PCA(n_components=3)
reduced = pca.fit_transform(vectors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2])
plt.title("3D PCA of Embeddings")
plt.savefig(os.path.join(output_dir, "pca_3d_plot.png"))
plt.close()

# Cosine Similarity Heatmap
vectors_np = vectors.numpy()
sim_matrix = cosine_similarity(vectors_np)

plt.figure()
sns.heatmap(sim_matrix, cmap="viridis")
plt.title("Cosine Similarity Between Clinical Texts")
plt.savefig(os.path.join(output_dir, "cosine_similarity_heatmap.png"))
plt.close()

# Cosine Similarity Clustered Map
sns.clustermap(sim_matrix, cmap="viridis", figsize=(12, 10),
               xticklabels=[os.path.basename(p) for p in embedding_dict.keys()],
               yticklabels=[os.path.basename(p) for p in embedding_dict.keys()])
plt.title("Cosine Similarity (Clustered)")
plt.savefig(os.path.join(output_dir, "cosine_similarity_clustermap.png"))
plt.close()

# UMAP with Thumbnails
fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0)

for i, (x0, y0) in enumerate(reduced[:, :2]):
    try:
        img = Image.open(list(embedding_dict.keys())[i]).resize((32, 32))
        imagebox = OffsetImage(img, zoom=0.6)
        ab = AnnotationBbox(imagebox, (x0, y0), frameon=False)
        ax.add_artist(ab)
    except:
        continue

plt.title("UMAP with Image Thumbnails")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "umap_image_thumbnails.png"))
plt.close()
