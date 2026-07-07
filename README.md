# Web Page Genre Classification

This project implements a multi-label genre classification system for web pages using **multiple Hugginface models** . It supports both single-model evaluation and an **averaging ensemble approach** to improve prediction accuracy across 9 distinct genre categories.

## 📋 Project Overview

The model classifies web page content into one of the following genres:
1. Research
2. News
3. Report
4. Project
5. Product
6. People
7. Event
8. Grant
9. Other/Uncategorized (Label 0)

## 🤖 Trained Models 

For this project we've used ensebmling to achieve improved results by averaging the outputs of multiple fine-tuned models. Models fine-tuned for this project are as follows:

- google-bert/bert-large-uncased
- distilbert/distilbert-base-uncased
- microsoft/deberta-large
- FacebookAI/xlm-roberta-large
- google/bigbird-roberta-large


### Key Features
*   **Model:** Fine-tuned 5 different transformers language models for sequence classification.
*   **Strategy:** Supports both single-checkpoint evaluation and **Model Ensemble** averaging.
*   **Efficiency:** Uses partial freezing of transformer layers to balance training speed and performance.
*   **Metrics:** Comprehensive evaluation using Precision, Recall, F1-Score, and Accuracy.

## 🛠️ Dependencies

Ensure you have the following libraries installed:

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib tqdm openpyxl
```

## 📂 Data Structure

The project expects the following files in the root directory:

| File | Description |
| :--- | :--- |
| `genre_final.xlsx` | Contains URLs and ground truth labels. |
| `page_data.xlsx` | Contains the raw text content of the web pages. |
| `genre_test_ids_new.npy` | A NumPy array containing the IDs of pages reserved for testing. |
| `Genre_Funcs.py` | Custom module containing preprocessing, dataset configuration, and training/evaluation logic. |

## ⚙️ Configuration

Key parameters can be adjusted at the top of the main script:

*   `train`: Set to `True` to fine-tune the model from scratch.
*   `test`: Set to `True` to run evaluation on the test set.
*   `ensemble`: Set to `True` to use multiple checkpoints for averaged predictions. 'test' must be set to `True` as well.
*   `fine_type`: Identifier for the model variant (e.g., `'distil'`).
*   `max_len`: Maximum sequence length (default: 512).

## 🚀 Usage

### 1. Training (Optional)
To fine-tune the model, set `train = True` in the script. The code will:
1.  Load and preprocess the data using hf tokenizer (e.g., `DistilBERT`).
2.  Freeze most of its layers, training only the last two transformer layers and the classifier head.
3.  Save checkpoints to `./ensemble_models/`.

### 2. Evaluation & Inference
To evaluate the model, set `test = True`.

**Single Model Evaluation:**
Set `ensemble = False`. The script will load the specified checkpoint from `./ensemble_models/{fine_type}_genre_chkp/` and run predictions.

**Ensemble Evaluation:**
Set `ensemble = True`. The script will:
1.  Scan `./ensemble_models/` for available checkpoints.
2.  Load the available checkpoints.
3.  Average the predictions to produce a final classification.

### 3. Running the Script
```bash
python genre_main.py
```

## 📊 Output Metrics

The script outputs a detailed classification report including:
*   **Precision (Macro)**
*   **Recall (Macro)**
*   **F1-Score (Macro)**
*   **Accuracy**

## 📝 Notes
*   **GPU Support:** The code is configured to run on CUDA. Ensure you have a compatible GPU and drivers installed.
*   **Data Preprocessing:** The `Genre_Funcs.py` module handles tokenization and padding. Ensure this file is in the same directory as the main script.
*   **Test Split:** The test set is strictly defined by `genre_test_ids_new.npy` to ensure consistent evaluation across different runs.
