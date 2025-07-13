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
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis("off")
        plt.show()

unlabeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\Unlabeled"
labeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\labelled"

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

def plot_scatter(df):
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(df['WHO Grade'])), df['WHO Grade'].astype(str), alpha=0.6, color='blue')
    plt.xlabel("Sample Index")
    plt.ylabel("WHO Grade")
    plt.title("Scatter Plot of WHO Grades")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

import seaborn as sns

def plot_boxplot(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df['WHO Grade'])
    plt.title("Boxplot of WHO Grades")
    plt.show()


def plot_bar(df):
    grade_counts = df['WHO Grade'].value_counts()
    plt.figure(figsize=(8, 5))
    grade_counts.plot(kind='bar', color='orange', alpha=0.7)
    plt.xlabel("WHO Grade")
    plt.ylabel("Count")
    plt.title("Distribution of WHO Grades")
    plt.xticks(rotation=45)
    plt.grid(axis="y")
    plt.show()

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
    outliers, non_outliers = detect_outliers(df)

    plt.figure(figsize=(8, 5))
    plt.bar(["Non-Outliers", "Outliers"], [len(non_outliers), len(outliers)], color=['green', 'red'])
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("Outlier vs Non-Outlier Distribution")
    plt.show()

    print(f"Number of Non-Outliers: {len(non_outliers)}")
    print(f"Number of Outliers: {len(outliers)}")

csv_file_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\IPD_Brain.csv"
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
    labels = ['Matched', 'Unmatched']
    sizes = [len(matched), len(unmatched)]
    colors = ['#4CAF50', '#FF5733']
    explode = (0.1, 0)
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, startangle=140)
    plt.title('Labeled Images: Matched vs Unmatched')
    plt.show()

csv_file_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\IPD_Brain.csv"
df = pd.read_csv(csv_file_path)

labeled_folder_path = r"C:\Users\rohan\OneDrive\Desktop\labelled"
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
    grade_counts = df['WHO Grade'].value_counts()
    
    plt.figure(figsize=(8, 6))
    grade_counts.plot(kind='bar', color=['#4CAF50', '#FF5733', '#FFC107', '#2196F3'])
    plt.xlabel('Grade')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of Cases by Grade')
    plt.xticks(rotation=0)
    plt.show()

def plot_images_per_grade(dataset_dir):
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
    plt.show()

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

dataset_seg_path = r"C:\Users\rohan\OneDrive\Desktop\Codes\DatasetSeg"
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
