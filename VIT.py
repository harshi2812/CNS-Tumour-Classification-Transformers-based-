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
    save_dir = 'output_plots'
    os.makedirs(save_dir, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis("off")

        filename = os.path.splitext(os.path.basename(img_path))[0] + "_plot.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


unlabeled_folder_path = "/home/madhavimathur/Datasets/Unlabeled"
labeled_folder_path = "/home/madhavimathur/Datasets/labelled"

unlabeled_images = get_unlabeled_images(unlabeled_folder_path)
labeled_images = get_labeled_images(labeled_folder_path)

print("Unlabeled Image Paths:")
print(unlabeled_images)

print("\nLabeled Image Paths:")
print(labeled_images)

import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    if 'WHO Grade' not in df.columns:
        raise ValueError("Column 'WHO Grade' not found in CSV file.")
    return df

import os

def plot_scatter(df):
    save_dir = "output_plots"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(df['WHO Grade'])), df['WHO Grade'].astype(str), alpha=0.6, color='blue')
    plt.xlabel("Sample Index")
    plt.ylabel("WHO Grade")
    plt.title("Scatter Plot of WHO Grades")
    plt.xticks(rotation=45)
    plt.grid()

    save_path = os.path.join(save_dir, "scatter_plot_who_grades.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {os.path.abspath(save_path)}")


import seaborn as sns

def plot_boxplot(df):
    save_dir = "output_plots"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['WHO Grade'])
    plt.title("Boxplot of WHO Grades")

    save_path = os.path.join(save_dir, "boxplot_who_grades.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Boxplot saved to: {os.path.abspath(save_path)}")


def plot_bar(df):
    save_dir = "output_plots"
    os.makedirs(save_dir, exist_ok=True)

    grade_counts = df['WHO Grade'].value_counts()
    plt.figure(figsize=(8, 5))
    grade_counts.plot(kind='bar', color='orange', alpha=0.7)
    plt.xlabel("WHO Grade")
    plt.ylabel("Count")
    plt.title("Distribution of WHO Grades")
    plt.xticks(rotation=45)
    plt.grid(axis="y")

    save_path = os.path.join(save_dir, "barplot_grade_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Bar plot saved to: {os.path.abspath(save_path)}")

def detect_outliers(df):
    grades = df['WHO Grade']
    q1 = grades.quantile(0.25)
    q3 = grades.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(grades < lower_bound) | (grades > upper_bound)]
    non_outliers = df[(grades >= lower_bound) & (grades <= upper_bound)]

    return outliers, non_outliers

def check_duplicates(df):
    duplicate_count = df.duplicated().sum()
    print(f"Number of Duplicate Rows: {duplicate_count}")
    return duplicate_count

def plot_outliers(df):
    save_dir = "output_plots"
    os.makedirs(save_dir, exist_ok=True)

    outliers, non_outliers = detect_outliers(df)

    plt.figure(figsize=(8, 5))
    plt.bar(["Non-Outliers", "Outliers"], [len(non_outliers), len(outliers)], color=['green', 'red'])
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("Outlier vs Non-Outlier Distribution")

    save_path = os.path.join(save_dir, "outlier_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Outlier plot saved to: {os.path.abspath(save_path)}")

    print(f"Number of Non-Outliers: {len(non_outliers)}")
    print(f"Number of Outliers: {len(outliers)}")


csv_file_path = "/home/madhavimathur/Datasets/IPD_Brain.csv"
df = load_data(csv_file_path)

plot_scatter(df)
plot_bar(df)
plot_outliers(df)
check_missing_values(df)
check_duplicates(df)
plot_boxplot(df)


import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_case_number(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def check_labeled_images_in_csv(labeled_image_paths, df):
    image_case_numbers = {extract_case_number(path) for path in labeled_image_paths}
    csv_case_numbers = set(df['Case Number'].astype(str))

    matched_cases = image_case_numbers.intersection(csv_case_numbers)
    unmatched_cases = image_case_numbers.difference(csv_case_numbers)

    if unmatched_cases:
        print("Warning: The following labeled images are missing from the CSV:")
        for case in unmatched_cases:
            print(case)
    else:
        print("All labeled images are present in the CSV.")
    
    return matched_cases, unmatched_cases

def plot_pie_chart(matched, unmatched):
    save_dir = "output_plots"
    os.makedirs(save_dir, exist_ok=True)

    labels = ['Matched', 'Unmatched']
    sizes = [len(matched), len(unmatched)]
    colors = ['#4CAF50', '#FF5733']
    explode = (0.1, 0)
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=140)
    plt.title('Labeled Images: Matched vs Unmatched')

    save_path = os.path.join(save_dir, "labeled_images_pie_chart.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Pie chart saved to: {os.path.abspath(save_path)}")


csv_file_path = "/home/madhavimathur/Datasets/IPD_Brain.csv"
df = pd.read_csv(csv_file_path)

labeled_folder_path = "/home/madhavimathur/Datasets/labelled"
labeled_images = [os.path.join(labeled_folder_path, file) for file in os.listdir(labeled_folder_path) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

matched, unmatched = check_labeled_images_in_csv(labeled_images, df)
plot_pie_chart(matched, unmatched)


import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_case_number(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def check_labeled_images_in_csv(labeled_image_paths, df):
    image_case_numbers = {extract_case_number(path) for path in labeled_image_paths}
    csv_case_numbers = set(df['Case Number'].astype(str))

    matched_cases = image_case_numbers.intersection(csv_case_numbers)
    unmatched_cases = image_case_numbers.difference(csv_case_numbers)

    if unmatched_cases:
        print("Warning: The following labeled images are missing from the CSV:")
        for case in unmatched_cases:
            print(case)
    else:
        print("All labeled images are present in the CSV.")
    
    return matched_cases, unmatched_cases

def plot_grade_distribution(df):
    save_dir = "output_plots"
    os.makedirs(save_dir, exist_ok=True)

    grade_counts = df['WHO Grade'].value_counts()
    plt.figure(figsize=(8, 6))
    grade_counts.plot(kind='bar', color=['#4CAF50', '#FF5733', '#FFC107', '#2196F3'])
    plt.xlabel('Grade')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of Cases by Grade')
    plt.xticks(rotation=0)

    save_path = os.path.join(save_dir, "grade_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Grade distribution plot saved to: {os.path.abspath(save_path)}")


def plot_images_per_grade(dataset_dir):
    save_dir = "output_plots"
    os.makedirs(save_dir, exist_ok=True)

    grade_counts = {}
    for grade_folder in os.listdir(dataset_dir):
        grade_path = os.path.join(dataset_dir, grade_folder)
        if os.path.isdir(grade_path):
            grade_counts[grade_folder] = len(os.listdir(grade_path))

    plt.figure(figsize=(8, 6))
    plt.bar(grade_counts.keys(), grade_counts.values(), color=['#4CAF50', '#FF5733', '#FFC107', '#2196F3'])
    plt.xlabel('Grade')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Grade')
    plt.xticks(rotation=0)

    save_path = os.path.join(save_dir, "images_per_grade.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Images per grade plot saved to: {os.path.abspath(save_path)}")

def organize_images_by_grade(labeled_image_paths, df, dataset_dir):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    for _, row in df.iterrows():
        case_number = str(row['Case Number'])
        grade = f"Grade {row['WHO Grade']}"
        grade_folder = os.path.join(dataset_dir, grade)
        
        if not os.path.exists(grade_folder):
            os.makedirs(grade_folder)
        
        for image_path in labeled_image_paths:
            if extract_case_number(image_path) == case_number:
                new_path = os.path.join(grade_folder, os.path.basename(image_path))
                os.rename(image_path, new_path)


df = pd.read_csv(csv_file_path)
labeled_images = [os.path.join(labeled_folder_path, file) for file in os.listdir(labeled_folder_path) if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

matched, unmatched = check_labeled_images_in_csv(labeled_images, df)
plot_grade_distribution(df)

dataset_seg_path = "/home/madhavimathur/Dataseg"
organize_images_by_grade(labeled_images, df, dataset_seg_path)
plot_images_per_grade(dataset_seg_path)

import pandas as pd

def check_grade_counts(df):
    if 'WHO Grade' not in df.columns:
        print("Error: 'WHO Grade' column not found in the DataFrame.")
        return None

    grade_counts = df['WHO Grade'].value_counts().to_dict()
    
    if 1 in grade_counts:
        print(f"Yes, there are {grade_counts[1]} cases of Grade 1.")
    else:
        print("No, there are 0 cases of Grade 1.")

    return grade_counts  

df = pd.read_csv(csv_file_path)
grade_distribution = check_grade_counts(df)

if grade_distribution:
    print("Grade counts:", grade_distribution)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

file_path = csv_file_path
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
df.fillna("Unknown", inplace=True)

print(df.head())
print(df.describe())

for col in ["WHO Grade", "SITE", "Subtype", "IDH1R132H", "ATRX", "p53"]:
    print(f"{col}:\n{df[col].value_counts()}\n")

plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=15, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(y=df["ki67(in %)"])
plt.title("Ki67% Distribution")
plt.ylabel("Ki67 (in %)")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x=df["WHO Grade"], palette="coolwarm")
plt.title("WHO Grade Distribution")
plt.xlabel("WHO Grade")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df[["Age", "ki67(in %)"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter
from mpl_toolkits.mplot3d import Axes3D

file_path = csv_file_path
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
df.fillna("Unknown", inplace=True)

print(df.head())
print(df.describe())

for col in ["WHO Grade", "SITE", "Subtype", "IDH1R132H", "ATRX", "p53"]:
    print(f"{col}:\n{df[col].value_counts()}\n")

plt.figure(figsize=(8,5))
sns.violinplot(x=df["Subtype"], y=df["Age"], palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Age Distribution Across Tumor Subtypes")
plt.show()

sns.pairplot(df, hue="WHO Grade", diag_kind="kde")
plt.show()

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["Age"], df["ki67(in %)"] , df["WHO Grade"], c=df["WHO Grade"], cmap='coolwarm')
ax.set_xlabel("Age")
ax.set_ylabel("Ki67%")
ax.set_zlabel("WHO Grade")
plt.title("3D Scatter Plot of Tumor Characteristics")
plt.show()

if "Survival_Months" in df.columns and "Survival_Status" in df.columns:
    kmf = KaplanMeierFitter()
    kmf.fit(df["Survival_Months"], event_observed=df["Survival_Status"])
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Months")
    plt.ylabel("Survival Probability")
    plt.show()

X = df[["Age", "ki67(in %)"]].dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["Age"], y=df["ki67(in %)"] , hue=df["Cluster"], palette="viridis")
plt.title("KMeans Clustering of Age and Ki67%")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(pd.crosstab(df["IDH1R132H"], df["ATRX"], normalize='index'), annot=True, cmap="coolwarm")
plt.title("Mutational Co-occurrence Heatmap")
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim

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

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

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
        label = self.labels[idx]
        return img, label

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
    save_dir = "output_plots"
    os.makedirs(save_dir, exist_ok=True)

    for path, grade in file_to_grade_dict.items():
        try:
            image = Image.open(path)
            plt.imshow(image)
            plt.title(f"Grade {grade}")
            plt.axis('off')

            # Create a safe filename based on the path and grade
            filename = os.path.basename(path).split('.')[0]
            save_path = os.path.join(save_dir, f"{filename}_grade_{grade}.png")
            plt.savefig(save_path)
            plt.close()

            print(f"Image with grade {grade} saved to: {os.path.abspath(save_path)}")
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

augmented_folder_path = "/home/madhavimathur/augmented"

if os.path.exists(augmented_folder_path):
    shutil.rmtree(augmented_folder_path)
os.makedirs(augmented_folder_path)

grade_to_files = defaultdict(list)
for file, grade in file_to_grade_dict.items():
    grade_to_files[grade].append(file)

grade_counts = {grade: len(files) for grade, files in grade_to_files.items()}
max_count = max(grade_counts.values())

augmented_file_to_grade_dict = dict(file_to_grade_dict)

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

            aug_path = os.path.join(augmented_folder_path, f"{name}_aug{i}_{aug_type}{ext}")
            aug_image.save(aug_path)
            augmented_file_to_grade_dict[aug_path] = grade

            plt.imshow(aug_image)
            plt.title(f"Grade {grade} - Aug Type: {aug_type}")
            plt.axis('off')
            plt.show()

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

dataset = ImageDataset(augmented_file_to_grade_dict, transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2) 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768,
    num_heads=12,
    depth=6,
    mlp_dim=3072,
    num_classes=4
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
epochs = 10

train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, marker='o', color='blue')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accuracies, marker='o', color='green')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()
