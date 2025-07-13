🧠 Multi-Modal Brain Tumor Classification using Vision Transformers and Clinical Metadata

Harshil Handoo

Advisor: Dr. Tav Pritesh Sethi Institution: (IIIT-Delhi)

📌 Abstract

Accurate classification of brain tumors in histopathological scans remains a challenge due to high intra-class variability, complex tumor morphology, and the limited availability of annotated datasets. We propose a novel multi-modal learning framework that combines spatial features from H&E-stained images via custom-designed Vision Transformers (ViT) with textual encodings derived from structured clinical metadata using LSTM and BioBERT. The two modalities are aligned through contrastive learning to enable robust cross-modal representation learning, contributing to an integrated PLIP-style architecture for tumor grading and generation.

🎯 Problem Statement

Despite significant advancements, current tumor classification techniques:

Struggle with intra-class variability and limited annotated datasets.

Often ignore valuable clinical metadata that could improve diagnostic accuracy.

Lack generalization across tumor subtypes and patient demographics.

🧪 Methodology

🧼 Step 1: Data Preprocessing

Cleaned and normalized structured metadata.

Removed outliers, imputed missing values.

Aligned image IDs with corresponding patient records.

🖼️ Step 2: Vision Transformer (ViT)

Custom ViT model trained on histopathological image patches.

Enhanced representation learning for spatially rich image regions.

ViT Mathematical Representation:

z₀ = [x_class; x₁ᵀᴇ; ... ; xₙᵀᴇ] + E_pos  
z_l = MLP(MSA(z_{l−1}) + z_{l−1}) + MSA(z_{l−1})

📄 Step 3 & 4: Textual Encoders – LSTM & BioBERT

Fine-tuned LSTM and BioBERT on clinical metadata like WHO Grade, Mutation Status, Age, Subtype, etc.

LSTM Representation:

h_t = LSTM(x_t, h_{t−1})  
y = Softmax(W_h·h_T + b)

BioBERT Embedding:

H = BERT(T) = [CLS; e₁; ... ; eₙ]  
y = Softmax(W_cls · H[CLS] + b)

⚖️ Step 5: Cross-Modal Fusion using PLIP + Contrastive Learning

Text and image encoders trained jointly using contrastive loss to bring matching modalities closer in embedding space.

Contrastive Loss Formula:

L = −log ( exp(sim(zᵢ, zᵭ)/τ) / ∑ₖ ⧉[k ≠ i] exp(sim(zᵢ, zₖ)/τ) )

🧬 Step 6: Augmentation & Evaluation

Applied image augmentation techniques (flipping, color jittering) and upsampled underrepresented WHO grades.

Dimensionality reduction (PCA, t-SNE) showed well-separated tumor grade clusters.

Evaluated classification accuracy, interpretability, and embedding alignment.

📊 Results & Analysis

Component

Approach

Performance

Visual Encoder

Vision Transformer / Swin Transformer

High-grade class separation

Text Encoder

LSTM / BioBERT

Accurate metadata embeddings

Fusion

PLIP-style with Contrastive Loss

Cross-modal consistency

Output

WHO Grade Classification

Improved accuracy & robustness

🧹 Model Architecture Overview

[ Histopathology Image ] --> [ Vision Transformer ]
                                ↓
                            [ Contrastive Fusion ] <--> [ LSTM / BioBERT ]
                                                        ↑
                                  [ Clinical Metadata (CSV) ]

📀 Folder Structure

/images                → Histopathology image inputs
/clinical_data         → Metadata CSVs
/models                → ViT, LSTM, BioBERT checkpoints
/results               → Visualizations & classification results
/utils                 → Data loaders and preprocessors

👉 Conclusion

Developed a self-designed ViT architecture fine-tuned for histopathological brain tumor scans.

Built dual textual encoders (LSTM + BioBERT) for robust metadata understanding.

Integrated both streams via PLIP-style contrastive learning, enabling multi-modal tumor classification and generation.

🚀 Future Directions

Fine-tune on larger datasets (e.g., AIIMS or TCGA) for improved generalizability.

Expand into image-to-text and text-to-image generation with PLIP backbone.

Explore advanced combinations like Swin Transformer + BioBERT for improved performance.

💡 Technologies Used

Python · PyTorch · Transformers · Matplotlib · Seaborn

HuggingFace Transformers · OpenCV · BioBERT · PLIP

Scikit-learn · ViT & Swin Transformer architectures

👨‍🏫 Acknowledgements

This project was conducted under the guidance of Dr. Tav Pritesh Sethi at IIIT-Delhi.

📬 Contact

For any questions or collaboration inquiries, feel free to reach out:📧 harshil.handoo@iiitd.ac.in
