from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import os
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import zipfile
import os
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import Pool, cpu_count
import os
import cv2
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image

plot_output_dir = "/home/madhavimathur/plots_output_swim"
os.makedirs(plot_output_dir, exist_ok=True)



def get_unlabeled_images(unlabeled_folder):
    unlabeled_image_paths = []

    for file in os.listdir(unlabeled_folder):
        if file.endswith(".png"):
            img_path = os.path.join(unlabeled_folder, file)
            unlabeled_image_paths.append(img_path)

    return unlabeled_image_paths

def get_labeled_images(labeled_folder):
    labeled_image_paths = []

    for file in os.listdir(labeled_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            labeled_image_paths.append(os.path.join(labeled_folder, file))

    return labeled_image_paths

def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing Values in Dataset:")
    print(missing_values[missing_values > 0])
    return missing_values


def show_unlabeled_images(image_paths):
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_output_dir, "plot_1.png"))
        plt.close()


unlabeled_folder_path = "/home/madhavimathur/Datasets/Unlabeled"
labeled_folder_path = "/home/madhavimathur/Datasets/labelled"

unlabeled_images = get_unlabeled_images(unlabeled_folder_path)
labeled_images = get_labeled_images(labeled_folder_path)


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

modelText = TextEncoder(vocab_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelText.to(device)
embedding_dict = {}

modelText.eval()
with torch.no_grad():
    for paths, encoded_texts in loader:
        encoded_texts = encoded_texts.to(device)
        embeddings = modelText(encoded_texts)
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
    
   
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def get_file_to_grade_dict(labeled_folder_path, csv_path):
    def normalize_case_name(case_name):
        return case_name.strip().replace('"', '').replace("’", "'").replace("\r", "").replace("\t", "")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    case_to_grade = {}

    for _, row in df.iterrows():
        grade = str(row.get("WHO Grade", "")).strip()
        raw_cases = str(row.get("Case Number", "")).strip().splitlines()
        for case in raw_cases:
            case = normalize_case_name(case)
            if case:
                case_to_grade[case] = grade

    file_to_grade = {}
    for file in os.listdir(labeled_folder_path):
        if file.lower().endswith((".png", ".jpg")):
            full_path = os.path.join(labeled_folder_path, file)
            name_without_ext = os.path.splitext(file)[0]
            matched_grade = case_to_grade.get(name_without_ext, None)
            if matched_grade is None:
                print(f"[!] Grade not found for: {file}")
                matched_grade = "Not Found"
            file_to_grade[full_path] = matched_grade

    return file_to_grade

def show_images_with_grades(file_to_grade_dict):
    for path, grade in file_to_grade_dict.items():
        try:
            image = Image.open(path)
            plt.imshow(image)
            plt.title(f"Grade {grade}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_output_dir, "grade.png"))
            plt.close()
        except Exception as e:
            print(f"[!] Could not display {path}: {e}")

labeled_folder_path = "/home/madhavimathur/Datasets/labelled"
csv_path = "/home/madhavimathur/Datasets/IPD_Brain.csv"

file_to_grade_dict = get_file_to_grade_dict(labeled_folder_path, csv_path)
print(file_to_grade_dict)
grade_count = {}

for grade in file_to_grade_dict.values():
    grade_count[grade] = grade_count.get(grade, 0) + 1

for grade, count in sorted(grade_count.items(), key=lambda x: int(x[0])):
    print(f"Grade {grade}: {count}")
import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW
from tqdm import tqdm
from timm import create_model
model = create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=4)
model.to(device)
print(model)

class ImageGradeDataset(Dataset):
    def __init__(self, file_to_grade_dict, transform=None):
        self.file_to_grade_dict = file_to_grade_dict
        self.transform = transform

    def __len__(self):
        return len(self.file_to_grade_dict)

    def __getitem__(self, idx):
        img_path = list(self.file_to_grade_dict.keys())[idx]
        grade = self.file_to_grade_dict[img_path]

        if grade == "Not Found":
            grade = -1
        else:
            try:
                grade = int(grade) - 1
            except ValueError:
                grade = -1

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        grade_description = f"The grade for this image is {grade + 1}" if grade != -1 else "Grade not available"
        return image, grade_description

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

class ImageGradeDataset(Dataset):
    def __init__(self, file_to_grade_dict, transform=None):
        self.paths, self.labels = [], []
        for path, grade in file_to_grade_dict.items():
            try:
                self.paths.append(path)
                self.labels.append(int(grade) - 1)
            except:
                continue
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

dataset = ImageGradeDataset(file_to_grade_dict, transform=transform)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
model.head = nn.Linear(model.head.in_features, 4)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

epochs = 200
for epoch in range(epochs):
    model.train()
    running_loss, correct = 0.0, 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / len(train_loader.dataset))

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(100 * val_correct / len(val_loader.dataset))

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Acc: {train_accuracies[-1]:.2f}% | Val Loss: {val_losses[-1]:.4f}, Acc: {val_accuracies[-1]:.2f}%")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, "train_validation_accuracy_loss.png"))
plt.close()

import torch
from torchvision import transforms
from PIL import Image
from timm import create_model

unlabeled_folder_path = '/home/madhavimathur/Datasets/Unlabeled'
unlabeled_images = get_unlabeled_images(unlabeled_folder_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('swin_base_patch4_window7_224', pretrained=True)
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for image_path in unlabeled_images:
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        if torch.isnan(output).any() or output.abs().sum() == 0:
            print(f'Invalid output for {image_path}')
        else:
            print(f'Successful inference: {image_path}')
    except Exception as e:
        print(f'Error processing {image_path}: {e}')