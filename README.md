# NLP Emotion Classification

## Project Overview
This project aims to classify text samples according to the primary emotion expressed, using machine learning techniques. The workflow includes data preprocessing, model training (including both traditional and deep learning models), evaluation, and analysis of results.

## Dataset Description
- The dataset is split into `train.txt`, `test.txt`, and `validation.txt` files (the latter for final evaluation).
- Each file contains lines with a text sample and its corresponding emotion label, separated by a semicolon (`;`).
- Emotion classes include: `anger`, `fear`, `joy`, `love`, `sadness`, and `surprise`.

## Setup Instructions
1. **Install dependencies:**
   - Python 3.8+
   - PyTorch (with CUDA support if available)
   - scikit-learn
   - pandas
   - matplotlib, seaborn
   - torchinfo (for model summary)
   - TensorFlow/Keras (for LSTM models)
   
   Example installation:
   ```bash
   pip install torch torchvision torchaudio scikit-learn pandas matplotlib seaborn torchinfo tensorflow
   ```
2. **Ensure the dataset files are in the `NLP_exam_emotions_dataset/` directory.**

## Model Training
- The project includes both a Fully Connected Neural Network (FCNN, implemented in PyTorch) and deep learning models (LSTM and Bidirectional LSTM, implemented in Keras/TensorFlow).
- **FCNN:**
  - Text data is vectorized using TF-IDF with up to 5000 features.
  - Labels are encoded as integers using scikit-learn's `LabelEncoder`.
  - Training and validation are performed using PyTorch `DataLoader` objects.
  - The best model (based on validation accuracy) is saved as `best_fcnn_model.pth`.
- **LSTM and Bidirectional LSTM:**
  - Text is tokenized and padded to a fixed length.
  - Labels are integer-encoded.
  - The LSTM model uses an Embedding layer, an LSTM (or Bidirectional LSTM) layer, and a Dense output layer with softmax activation.
  - Class weights and early stopping are used to address class imbalance and prevent overfitting.
  - Training and validation accuracy/loss are tracked for each epoch.

## Evaluation and Results
- After training, the models' performance is visualized using loss and accuracy curves.
- A confusion matrix is generated to analyze per-class performance and misclassifications.
- The Bidirectional LSTM achieved high overall accuracy (>90%), but the confusion matrix revealed that the model is biased toward the majority class ("joy"), with minority classes often misclassified. This highlights the importance of addressing class imbalance in emotion classification tasks.
- For detailed explanations, see:
  - [plot_explanation.md](plot_explanation.md) — for training/validation curves
  - [confusion_matrix_explanation.md](confusion_matrix_explanation.md) — for confusion matrix analysis

## How to Run
1. Run the provided Jupyter notebook or Python scripts to preprocess data, train the model (FCNN, LSTM, or Bidirectional LSTM), and evaluate results.
2. Use the plotting and analysis scripts to interpret model performance.

## Notes
- For best results, ensure your environment has access to a GPU.
- You can experiment with model architecture, hyperparameters, or more advanced NLP features for further improvements.
- The project demonstrates the effectiveness of deep learning for text emotion classification, while also highlighting the importance of balanced data and careful evaluation using confusion matrices and per-class metrics.

---

For questions or suggestions, please contact the project maintainer. 