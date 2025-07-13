import zipfile
import os
from PIL import Image
import matplotlib.pyplot as plt

zip_path = '/content/Dataset1.zip'
extract_path = '/content/Dataset1'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted files and folders:")
image_files = []
for root, dirs, files in os.walk(extract_path):
    for name in dirs:
        print("DIR :", os.path.join(root, name))
    for name in files:
        file_path = os.path.join(root, name)
        print("FILE:", file_path)
        if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_files.append(file_path)

for img_path in image_files:
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(os.path.basename(img_path))
    plt.axis('off')
    plt.show()

import pandas as pd

excel_path = '/content/Dataset1/Dataset1/TCIA-CPTAC-GBM_v16_20240708-nbia-digest.xlsx'

df = pd.read_excel(excel_path)

print("Column names in the Excel file:")
for col in df.columns:
    print(col)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/content/Dataset1/Dataset1/TCIA-CPTAC-GBM_v16_20240708-nbia-digest.xlsx'
df = pd.read_excel(file_path)

columns_to_drop = [
    'Patient Name', 'Phantom', 'Species Code', 'Series Instance UID', 
    'Study Instance UID', 'License URI', 'Collection URI'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

df['Patient Birth Date'] = pd.to_datetime(df['Patient Birth Date'], errors='coerce')
df['Study Date'] = pd.to_datetime(df['Study Date'], errors='coerce')
df['Series Date'] = pd.to_datetime(df['Series Date'], errors='coerce')
df['Date Released'] = pd.to_datetime(df['Date Released'], errors='coerce')

text_cols = ['Patient Sex', 'Ethnic Group', 'Species Description', 'Project',
             'Modality', 'Protocol Name', 'Series Description', 'Body Part Examined',
             'Manufacturer', 'Manufacturer Model Name', 'License Name', 'Third Party Analysis']
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
df[text_cols] = df[text_cols].fillna('unknown')

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

for col in ['Patient Sex', 'Ethnic Group', 'Modality', 'Body Part Examined']:
    plt.figure(figsize=(8, 4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/content/Dataset1/Dataset1/TCIA-CPTAC-GBM_v16_20240708-nbia-digest.xlsx'
df = pd.read_excel(file_path)

df['Patient Birth Date'] = pd.to_datetime(df['Patient Birth Date'], errors='coerce')
df['Study Date'] = pd.to_datetime(df['Study Date'], errors='coerce')
df['Series Date'] = pd.to_datetime(df['Series Date'], errors='coerce')
df['Date Released'] = pd.to_datetime(df['Date Released'], errors='coerce')

text_cols = ['Patient Sex', 'Ethnic Group', 'Species Description', 'Project',
             'Modality', 'Protocol Name', 'Series Description', 'Body Part Examined',
             'Manufacturer', 'Manufacturer Model Name', 'License Name', 'Third Party Analysis']
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
df[text_cols] = df[text_cols].fillna('unknown')

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

cat_vars = ['Patient Sex', 'Ethnic Group', 'Modality', 'Body Part Examined', 'Project']
for col in cat_vars:
    plt.figure(figsize=(8, 4))
    df[col].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], bins=30, kde=True, color='lightgreen')
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

df['Study Date Only'] = df['Study Date'].dt.to_period('M')
study_counts = df['Study Date Only'].value_counts().sort_index()
plt.figure(figsize=(10, 4))
study_counts.plot(kind='line', marker='o')
plt.title('Number of Studies Over Time')
plt.ylabel('Study Count')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='Modality', y='Image Count')
plt.title('Image Count by Modality')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Between Numerical Columns')
    plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

file_path = '/content/Dataset1/Dataset1/TCIA-CPTAC-GBM_v16_20240708-nbia-digest.xlsx'
df = pd.read_excel(file_path)

df['Patient Age'] = df['Patient Age'].astype(str).str.extract(r'(\d+)').astype(float)
df['Patient Birth Date'] = pd.to_datetime(df['Patient Birth Date'], errors='coerce')
df['Study Date'] = pd.to_datetime(df['Study Date'], errors='coerce')
df['Series Date'] = pd.to_datetime(df['Series Date'], errors='coerce')
df['Date Released'] = pd.to_datetime(df['Date Released'], errors='coerce')

text_cols = ['Patient Sex', 'Ethnic Group', 'Species Description', 'Project',
             'Modality', 'Protocol Name', 'Series Description', 'Body Part Examined',
             'Manufacturer', 'Manufacturer Model Name', 'License Name', 'Third Party Analysis']
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
df[text_cols] = df[text_cols].fillna('unknown')

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

agg_data = df.groupby('Modality')[['Image Count', 'File Size', 'Patient Age']].mean().dropna()
agg_data = agg_data.head(5)
categories = list(agg_data.columns)
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
for i, row in agg_data.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=i)
    ax.fill(angles, values, alpha=0.1)
ax.set_title("Radar Chart: Image Count, File Size, Age by Modality")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

sunburst_df = df[['Project', 'Modality', 'Body Part Examined']].dropna()
fig = px.sunburst(sunburst_df, path=['Project', 'Modality', 'Body Part Examined'],
                  title="Sunburst: Project > Modality > Body Part Examined")
fig.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='Modality', y='Image Count', inner=None, color='lightblue')
sns.swarmplot(data=df, x='Modality', y='Image Count', color='black', size=3)
plt.title('Violin + Swarm: Image Count by Modality')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='Modality', y='Patient Age', inner=None, color='lightgreen')
sns.swarmplot(data=df, x='Modality', y='Patient Age', color='black', size=3)
plt.title('Violin + Swarm: Patient Age by Modality')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

g = sns.FacetGrid(df, col="Modality", col_wrap=3, height=3.5)
g.map(sns.histplot, "Image Count", bins=20, color='orchid')
plt.suptitle("Facet Grid: Image Count per Modality", y=1.02)
plt.tight_layout()
plt.show()
