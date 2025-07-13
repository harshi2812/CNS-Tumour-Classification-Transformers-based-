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
model = CLIPModel.from_pretrained("vinid/plip")
processor = CLIPProcessor.from_pretrained("vinid/plip")

plot_output_dir = "/home/madhavimathur/centralPipeline_plots"
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
        plt.show()

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
    
print(model)    
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

dataset = ImageGradeDataset(file_to_grade_dict, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = CLIPModel.from_pretrained("vinid/plip")
processor = CLIPProcessor.from_pretrained("vinid/plip")

optimizer = AdamW(model.parameters(), lr=5e-6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

epochs = 1                      # epochs = 1300
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    loop = tqdm(dataloader, leave=True)
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for images, grade_descriptions in loop:
        images = images.to(device)
        inputs = processor(text=grade_descriptions, images=images, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        
        labels = torch.arange(len(logits_per_image), device=device)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        _, predicted = torch.max(logits_per_image, dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += len(labels)
        
        loop.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(loop)
    accuracy = 100 * correct_predictions / total_predictions
    
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

epochs_range = range(1, epochs+1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label="Training Accuracy", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy vs Epochs")
plt.grid(True)


plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, "train_validation_accuracy_loss.png"))
plt.close()

model.save_pretrained("trained_clip_model")
prompts = {
    1: "Image of a grade 1 tumor",
    2: "Image of a grade 2 tumor",
    3: "Image of a grade 3 tumor",
    4: "Image of a grade 4 tumor",
}

'''grade_prompts = [prompts[grade] for grade in range(1, 5)]
print(grade_prompts)



for grade, grade_description in file_to_grade_dict.items():
    prompt = prompts.get(grade, "No description available for this grade")
    inputs = processor(text=prompt, images=labeled_images, return_tensors="pt", padding=True)'''


from diffusers import StableDiffusionPipeline

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to(device)

# Fix: override safety checker properly
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

generated_image_output_dir = os.path.join(plot_output_dir, "generated_from_prompts")
os.makedirs(generated_image_output_dir, exist_ok=True)

for img_path, grade_str in file_to_grade_dict.items():
    if grade_str == "Not Found":
        continue
    try:
        grade_int = int(grade_str)
        prompt = prompts.get(grade_int, "No description available for this grade")
        
        print(f"[INFO] Generating image for prompt: {prompt}")
        
        # Generate image using Stable Diffusion
        generated_image = pipe(prompt).images[0]
        
        # Save the generated image
        base_name = os.path.basename(img_path)
        save_path = os.path.join(generated_image_output_dir, f"generated_{base_name}")
        generated_image.save(save_path)

        print(f"[SAVED] Generated image saved to: {save_path}\n")

    except Exception as e:
        print(f"[!] Error processing {img_path}: {e}")

print("✅ Finished image generation from prompts!")
