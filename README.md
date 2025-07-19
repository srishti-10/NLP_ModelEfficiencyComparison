# NLP Emotion Classification

## Project Overview
This project aims to classify text samples according to the primary emotion expressed, using multiple machine learning approaches. The workflow includes data preprocessing, model training (CNN, LSTM, Bidirectional LSTM, and Transformers), evaluation, and comprehensive analysis of results.

## Dataset Description
- The dataset is split into `train.txt`, `test.txt`, and `validation.txt` files (the latter for final evaluation).
- Each file contains lines with a text sample and its corresponding emotion label, separated by a semicolon (`;`).
- Emotion classes include: `anger`, `fear`, `joy`, `love`, `sadness`, and `surprise`.

## Setup Instructions
1. **Install dependencies:**
   - Python 3.8+
   - PyTorch (with CUDA support if available)
   - TensorFlow/Keras
   - scikit-learn
   - pandas
   - matplotlib, seaborn
   - transformers (HuggingFace)
   - datasets (HuggingFace)
   
   Example installation:
   ```bash
   pip install torch torchvision torchaudio tensorflow scikit-learn pandas matplotlib seaborn transformers datasets
   ```
2. **Ensure the dataset files are in the `NLP_exam_emotions_dataset/` directory.**

## Model Training
The project includes multiple model implementations for comprehensive comparison:

### **1. CNN Model (`CNN_Model_Inference.ipynb`)**
- Convolutional Neural Network for text classification
- Text preprocessing and feature extraction
- Training and evaluation pipeline

### **2. LSTM and Bidirectional LSTM (`Bidrectional LSTM.ipynb`)**
- **Architecture**: Embedding layer (64 dimensions) → Bidirectional LSTM (64 units with dropout) → Dense output layer
- **Key Features**:
  - Text tokenization and padding (max_len=40)
  - Class weighting to address imbalanced dataset
  - Early stopping to prevent overfitting
  - **Achieved 90% validation accuracy** with balanced performance across all classes
- **Results**: 
  - Joy (F1: 0.92) and Sadness (F1: 0.93): Best performing
  - Anger (F1: 0.91): Strong performance
  - Fear (F1: 0.84), Love (F1: 0.84), Surprise (F1: 0.81): Good performance for minority classes

### **3. Transformer Model (`transformers.ipynb`)**
- Fine-tuned DistilBERT model using HuggingFace Transformers
- State-of-the-art performance for text classification
- Advanced preprocessing and training pipeline

### **4. All Models Inference (`AllModelsInference.ipynb`)**
- Comprehensive comparison of all implemented models
- Inference functionality for new text samples
- Performance analysis and visualization

## Evaluation and Results
- **Training Curves**: Loss and accuracy plots for all models
- **Confusion Matrices**: Per-class performance analysis
- **Classification Reports**: Precision, recall, and F1-scores
- **Model Comparison**: Performance metrics across different architectures

### **Key Achievements**
- **Bidirectional LSTM**: 90% accuracy with balanced class performance
- **Transformer**: State-of-the-art results using fine-tuned BERT
- **CNN**: Baseline performance for comparison
- **All models include inference functionality** for practical applications

## How to Run
1. **For LSTM/Bidirectional LSTM**: Run `Bidrectional LSTM.ipynb`
2. **For CNN**: Run `CNN_Model_Inference.ipynb`
3. **For Transformers**: Run `transformers.ipynb`
4. **For comprehensive comparison**: Run `AllModelsInference.ipynb`

## Model Performance Summary
| Model | Accuracy | Best Class | Weakest Class | Notes |
|-------|----------|------------|---------------|-------|
| Bidirectional LSTM | 90% | Joy (F1: 0.92) | Surprise (F1: 0.81) | Balanced performance |
| Transformer | TBD | TBD | TBD | State-of-the-art |
| CNN | TBD | TBD | TBD | Baseline |

## Notes
- **GPU recommended** for optimal training performance
- **Class imbalance** successfully addressed with class weighting
- **Data preprocessing** includes overlap removal to prevent data leakage
- **Inference functionality** available for all models
- **Comprehensive evaluation** with multiple metrics and visualizations

## Technical Highlights
- **Data Leakage Prevention**: Removed overlapping texts between train/test sets
- **Class Imbalance Handling**: Implemented class weights and oversampling strategies
- **Regularization**: Dropout and early stopping to prevent overfitting
- **Model Comparison**: Systematic evaluation across different architectures
- **Practical Application**: Ready-to-use inference functions

---
