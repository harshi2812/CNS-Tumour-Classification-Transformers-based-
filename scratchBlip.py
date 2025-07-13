import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
from datasets import load_dataset
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, TrainingArguments, Trainer
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import torch
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
import pandas as pd
import json

df = pd.read_csv("/home/madhavimathur/Datasets/IPD_Brain.csv")

def make_caption(row):
    return f"Age: {row['Age']}, Sex: {row['Sex']}, Diagnosis: {row['Diagnosis']}, Grade: {row['WHO Grade']}, Site: {row['SITE']}, Subtype: {row['Subtype']}, Markers - IDH1: {row['IDH1R132H']}, ATRX: {row['ATRX']}, p53: {row['p53']}"

def to_coco_format(df, output):
    data = {"images": [], "annotations": []}
    for idx, row in df.iterrows():
        image_id = idx
        image_path = f"images/{row['Case Number']}.jpg"
        caption = make_caption(row)
        data["images"].append({
            "id": image_id,
            "file_name": image_path
        })
        data["annotations"].append({
            "id": idx,
            "image_id": image_id,
            "caption": caption
        })
    with open(output, 'w') as f:
        json.dump(data, f, indent=4)

def to_list_dict_format(df, output):
    all_data = []
    for _, row in df.iterrows():
        image_path = f"images/{row['Case Number']}.jpg"
        caption = make_caption(row)
        all_data.append({"image": image_path, "caption": caption})
    with open(output, 'w') as f:
        json.dump(all_data, f, indent=4)

def to_jsonl_format(df, output):
    with open(output, 'w') as f:
        for _, row in df.iterrows():
            image_path = f"images/{row['Case Number']}.jpg"
            caption = make_caption(row)
            json.dump({"image": image_path, "caption": caption}, f)
            f.write('\n')

def to_image_tabular_csv(df, output):
    df['image_path'] = df['Case Number'].apply(lambda x: f"images/{x}.jpg")
    df.to_csv(output, index=False)

to_coco_format(df, "coco_format.json")
to_list_dict_format(df, "huggingface_listdict.json")
to_jsonl_format(df, "transformer_data.jsonl")
to_image_tabular_csv(df, "full_data_with_image.csv")


dataset = load_dataset('json', data_files='huggingface_listdict.json')
dataset = dataset['train'].train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


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
    plot_output_dir = "blip_saved_plots"
    os.makedirs(plot_output_dir, exist_ok=True)

    for path, grade in file_to_grade_dict.items():
        try:
            image = Image.open(path)
            plt.imshow(image)
            plt.title(f"Grade {grade}")
            plt.axis('off')

            # Create a filename and save the plot
            base_name = os.path.basename(path)
            name, ext = os.path.splitext(base_name)
            plot_filename = f"{name}_grade{grade}_plot.png"
            plot_path = os.path.join(plot_output_dir, plot_filename)

            plt.savefig(plot_path, bbox_inches='tight')
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
import random
import shutil
from collections import defaultdict
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

file_to_grade_dict = get_file_to_grade_dict(labeled_folder_path, csv_path)

augmented_folder_path = "/home/madhavimathur/aug"

if os.path.exists(augmented_folder_path):
    shutil.rmtree(augmented_folder_path)
os.makedirs(augmented_folder_path)

grade_to_files = defaultdict(list)
for file, grade in file_to_grade_dict.items():
    grade_to_files[grade].append(file)

grade_counts = {grade: len(files) for grade, files in grade_to_files.items()}
max_count = max(grade_counts.values())

augmented_file_to_grade_dict = dict(file_to_grade_dict)

import os
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Define the directory to save plots
plot_output_dir = "blip_saved_plots"
os.makedirs(plot_output_dir, exist_ok=True)

for grade, files in grade_to_files.items():
    current_count = len(files)
    needed = max_count - current_count
    if needed <= 0:
        continue

    i = 0
    while i < needed:
        original_path = random.choice(files)
        try:
            image = Image.open(original_path)
            base_name = os.path.basename(original_path)
            name, ext = os.path.splitext(base_name)

            if i % 2 == 0:
                aug_image = ImageOps.mirror(image)
                aug_type = "flip"
            else:
                aug_image = image.rotate(90)
                aug_type = "rot90"

            # Save the augmented image
            aug_path = os.path.join(augmented_folder_path, f"{name}_aug{i}_{aug_type}{ext}")
            aug_image.save(aug_path)
            augmented_file_to_grade_dict[aug_path] = grade

            # Save the plot instead of displaying
            plt.imshow(aug_image)
            plt.title(f"Grade {grade} - Aug Type: {aug_type}")
            plt.axis('off')

            plot_filename = f"{name}_aug{i}_{aug_type}_plot.png"
            plot_path = os.path.join(plot_output_dir, plot_filename)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()  # Close the figure to prevent memory leaks

            i += 1

        except Exception as e:
            print(f"Error processing {original_path}: {e}")


grade_count = {}
for grade in augmented_file_to_grade_dict.values():
    grade_count[grade] = grade_count.get(grade, 0) + 1

for grade, count in sorted(grade_count.items(), key=lambda x: int(x[0])):
    print(f"Grade {grade}: {count}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])



import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from torchvision import transforms
from transformers import AutoTokenizer

class ImageGradeDataset(Dataset):
    def __init__(self, file_to_grade_dict, processor, transform=None):
        self.file_to_grade_dict = file_to_grade_dict
        self.processor = processor
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def __len__(self):
        return len(self.file_to_grade_dict)

    def __getitem__(self, idx):
        img_path, grade = list(self.file_to_grade_dict.items())[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        grade_input = self.tokenizer(grade, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        inputs = self.processor(images=image, return_tensors="pt")
        
        return inputs, grade_input
    
dataset = ImageGradeDataset(augmented_file_to_grade_dict, transform=transform)
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import LayerNorm

def check_dataloader(dataloader, name="Dataloader", num_batches=1):
    print(f"\n--- Checking {name} ---")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")

        if isinstance(batch, (list, tuple)):
            print(f"  Number of items in batch: {len(batch)}")
            for i, item in enumerate(batch):
                if isinstance(item, torch.Tensor):
                    print(f"  Item {i}:")
                    print(f"    Type   : Tensor")
                    print(f"    Shape  : {item.shape}")
                    print(f"    Dtype  : {item.dtype}")
                    print(f"    Numel  : {item.numel()}")
                    if item.dim() != 4:
                        print(f"  Warning: Tensor is not 4D (got {item.dim()}D)")
                else:
                    print(f"  Item {i}: Non-tensor | Type: {type(item)}")

        elif isinstance(batch, dict):
            print(f"  Number of keys in batch: {len(batch)}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  '{key}':")
                    print(f"    Type   : Tensor")
                    print(f"    Shape  : {value.shape}")
                    print(f"    Dtype  : {value.dtype}")
                    print(f"    Numel  : {value.numel()}")
                    if value.dim() == 4:
                        print(f"  '{key}' is a valid 4D tensor")
                    else:
                        print(f"  Warning: '{key}' is not 4D (got {value.dim()}D)")

                else:
                    print(f"  '{key}': Non-tensor | Type: {type(value)}")

            if 'label' in batch:
                print(f"  Labels: {batch['label']}")

        else:
            print(f"  Batch is not list/tuple/dict | Type: {type(batch)}")

        if batch_idx + 1 >= num_batches:
            break

        
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)    

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(x)

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embedding = torch.zeros(1, num_patches + 1, embed_dim)
        positions = torch.arange(num_patches + 1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        self.pos_embedding[:, :, 0::2] = torch.sin(positions * div_term)
        self.pos_embedding[:, :, 1::2] = torch.cos(positions * div_term)

    def forward(self, x):
        return x + self.pos_embedding.to(x.device)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, depth=12, mlp_dim=3072, num_classes=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.classifier(x)

class CrossAttentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, num_classes):
        super().__init__()
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim),
            num_layers=6
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, image_features, text_features):
        B = image_features.shape[0]
        memory = image_features.unsqueeze(0).repeat(B, 1, 1)
        tgt = text_features.unsqueeze(0).repeat(B, 1, 1)
        x = self.transformer(tgt, memory)
        x = self.norm(x)
        return self.classifier(x)

class BLIPModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, depth=12, mlp_dim=3072, num_classes=4):
        super().__init__()
        self.vit = VisionTransformer(img_size, patch_size, embed_dim=embed_dim, num_heads=num_heads, depth=depth, mlp_dim=mlp_dim, num_classes=num_classes)
        self.cross_attention_decoder = CrossAttentionDecoder(embed_dim, num_heads, mlp_dim, num_classes)

    def forward(self, img, text_features):
        image_features = self.vit(img)
        output = self.cross_attention_decoder(image_features, text_features)
        return output

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class ImageDataset(Dataset):
    def __init__(self, file_to_grade_dict, transform=None):
        self.file_to_grade_dict = {k: int(v) for k, v in file_to_grade_dict.items() if str(v).isdigit()}
        self.image_paths = list(self.file_to_grade_dict.keys())
        self.labels = list(self.file_to_grade_dict.values())
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img) if self.transform else img
        label = self.labels[idx] - 1 
        return img, label


dataset = ImageDataset(file_to_grade_dict, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
blip_model = BLIPModel(num_classes=4)


for img, text_features in data_loader:
    img = img.cuda()
    text_features = text_features.cuda() 
    output = blip_model(img, text_features)
    print(output.shape)


dataset = ImageDataset(file_to_grade_dict, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model = BLIPModel(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(blip_model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    blip_model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for img, text_features in data_loader:
        img, text_features = img.to(device), text_features.to(device)

        optimizer.zero_grad()
        output = blip_model(img, text_features)
        
        loss = criterion(output, text_features)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(output, 1)
        correct_preds += (predicted == text_features).sum().item()
        total_preds += text_features.size(0)
    
    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct_preds / total_preds * 100

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
