import zipfile
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import cv2

zip_file_path = "/content/Histopathology images.zip"
extract_to = "/content/Histopathology_images_extracted"

os.makedirs(extract_to, exist_ok=True)
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Extracted the zip file to: {extract_to}")
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, image_path

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = torch.nn.Identity()

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_folder = "/content/Histopathology_images_extracted/Histopathology images"
dataset = CustomImageDataset(image_folder, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

model = FeatureExtractor()
model.eval()

def extract_features(model, dataloader):
    features = []
    image_paths = []

    with torch.no_grad():
        for inputs, paths in dataloader:
            outputs = model(inputs)
            features.append(outputs.numpy())
            image_paths.extend(paths)

    features = np.concatenate(features, axis=0)
    return features, image_paths

features, image_paths = extract_features(model, dataloader)

print(f"Extracted features shape: {features.shape}")
print(f"Image paths: {image_paths[:5]}")

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(features)



agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(features)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(features)

def print_evaluation_metrics(labels, features):
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        print(f"Silhouette Score: {silhouette}")
        print(f"Davies-Bouldin Index: {davies_bouldin}")
        print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    else:
        print("Not enough clusters for DBSCAN metrics.")

print("\nKMeans Evaluation Metrics:")
print_evaluation_metrics(kmeans_labels, features)

print("\nAgglomerative Clustering Evaluation Metrics:")
print_evaluation_metrics(agg_labels, features)

print("\nDBSCAN Evaluation Metrics:")
print_evaluation_metrics(dbscan_labels, features)

print("\nKMeans Evaluation Metrics:")
print_evaluation_metrics(kmeans_labels, features)

print("\nAgglomerative Clustering Evaluation Metrics:")
print_evaluation_metrics(agg_labels, features)

print("\nDBSCAN Evaluation Metrics:")
print_evaluation_metrics(dbscan_labels, features)

plt.figure(figsize=(10, 6))
plt.scatter(features[:, 0], features[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('KMeans Clustering')
plt.xlabel('Feature Dimension 1')
plt.ylabel('Feature Dimension 2')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(features[:, 0], features[:, 1], c=agg_labels, cmap='viridis')
plt.title('Agglomerative Clustering')
plt.xlabel('Feature Dimension 1')
plt.ylabel('Feature Dimension 2')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(features[:, 0], features[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature Dimension 1')
plt.ylabel('Feature Dimension 2')
plt.show()




from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
import numpy as np

def perform_pca(features, max_components=50):
    n_samples, n_features = features.shape
    n_components = min(max_components, n_samples, n_features)
    pca = PCA(n_components=n_components, svd_solver='auto')
    reduced_features = pca.fit_transform(features)
    return reduced_features


reduced_features = perform_pca(features)

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans_labels = kmeans.fit_predict(reduced_features)

agg_clustering_ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels_ward = agg_clustering_ward.fit_predict(reduced_features)

agg_clustering_average = AgglomerativeClustering(n_clusters=3, linkage='average')
agg_labels_average = agg_clustering_average.fit_predict(reduced_features)

agg_clustering_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
agg_labels_complete = agg_clustering_complete.fit_predict(reduced_features)

def print_evaluation_metrics(labels, features):
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        print(f"Silhouette Score: {silhouette}")
        print(f"Davies-Bouldin Index: {davies_bouldin}")
        print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    else:
        print("Not enough clusters for DBSCAN metrics.")

print("\nKMeans Evaluation Metrics:")
print_evaluation_metrics(kmeans_labels, reduced_features)

print("\nAgglomerative Clustering Evaluation Metrics (ward):")
print_evaluation_metrics(agg_labels_ward, reduced_features)

print("\nAgglomerative Clustering Evaluation Metrics (average):")
print_evaluation_metrics(agg_labels_average, reduced_features)

print("\nAgglomerative Clustering Evaluation Metrics (complete):")
print_evaluation_metrics(agg_labels_complete, reduced_features)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('KMeans Clustering (PCA Reduced)')
plt.xlabel('Feature Dimension 1')
plt.ylabel('Feature Dimension 2')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=agg_labels_ward, cmap='viridis')
plt.title('Agglomerative Clustering (Ward Linkage)')
plt.xlabel('Feature Dimension 1')
plt.ylabel('Feature Dimension 2')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=agg_labels_average, cmap='viridis')
plt.title('Agglomerative Clustering (Average Linkage)')
plt.xlabel('Feature Dimension 1')
plt.ylabel('Feature Dimension 2')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=agg_labels_complete, cmap='viridis')
plt.title('Agglomerative Clustering (Complete Linkage)')
plt.xlabel('Feature Dimension 1')
plt.ylabel('Feature Dimension 2')

plt.show()


import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image
import os

def perform_pca(features):
    n_components = min(50, min(features.shape))
    pca = PCA(n_components=n_components, svd_solver='full')
    reduced_features = pca.fit_transform(features)
    return reduced_features

def load_images_from_folder(folder_path, target_size=(64, 64)):
    images = []
    image_paths = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            images.append(np.array(img).flatten())
            image_paths.append(img_path)
        except Exception as e:
            pass
    return np.array(images), image_paths

def predict_labels(image_folder, model):
    standard_dict = {}
    images, image_paths = load_images_from_folder(image_folder)
    reduced_features = perform_pca(images)
    labels = model.predict(reduced_features)
    
    for idx, image_path in enumerate(image_paths):
        standard_dict[image_path] = labels[idx]
    
    return standard_dict

image_folder = "/content/Histopathology_images_extracted/Histopathology images"
images, _ = load_images_from_folder(image_folder)
reduced_features = perform_pca(images)

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(reduced_features)

labels_dict = predict_labels(image_folder, kmeans)
print(labels_dict)

from transformers import DeiTForImageClassification, AutoImageProcessor
import torch
from torch import nn
from torch.optim import AdamW
from PIL import Image
import os
import matplotlib.pyplot as plt

model = DeiTForImageClassification.from_pretrained("facebook/deit-small-distilled-patch16-224")
feature_extractor = AutoImageProcessor.from_pretrained("facebook/deit-small-distilled-patch16-224")


def load_images_from_folder(folder_path, target_size=(224, 224)):
    images = []
    image_paths = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            images.append(img)
            image_paths.append(img_path)
        except Exception as e:
            pass
    return images, image_paths

def prepare_labels(image_paths, label_dict):
    labels = []
    for path in image_paths:
        labels.append(label_dict.get(path, -1))  
    return torch.tensor(labels, dtype=torch.long)

def fine_tune_convnext(image_folder, model, label_dict, epochs=40, batch_size=16, learning_rate=5e-5):
    images, image_paths = load_images_from_folder(image_folder)
    labels = prepare_labels(image_paths, label_dict)
    inputs = feature_extractor(images=images, return_tensors="pt")

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.train()

    loss_list = []  
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    plt.plot(range(epochs), loss_list, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

image_folder = "/content/Histopathology_images_extracted/Histopathology images"
fine_tune_convnext(image_folder, model, labels_dict)
print("Code ends")
