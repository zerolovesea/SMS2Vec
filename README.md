
# SMS2Vec


## Project Overview
SMS2Vec is an engineering-oriented pipeline framework for vectorizing text information such as SMS and emails. It is a practical solution for transforming raw text records into feature vectors for downstream machine learning tasks. The framework combines statistical word vectors (TF-IDF, Word2Vec), pretrained language model embeddings (RoBERTa, Qwen3, BGE-M3), statistical features, and sequential SMS signature features, enabling flexible feature engineering and supporting large-scale data processing in real-world scenarios.

Typical use cases include:
- User interest modeling for ad targeting and recommendation recall
- Binary classification of user text records (e.g., spam detection, intent prediction)
- Feature engineering from raw text and signature sequences for downstream models

## Environment
This project requires **Python 3.10**.

## Installation
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Data Format
Input CSV files should contain columns such as:

| id | message | sign | datetime | label |
|----|---------|------|----------|-------|
| ...| ...     | ...  | ...      | ...   |

ID can be duplicated, and each row contain an unique message record.

## Usage

### 1. Data Preprocessing
Edit `processing_config.yaml` to set preprocessing options (encryption, filtering, feature extraction, etc).
Run:
```bash
python main.py
```
This will process raw data (e.g., `data/raw/messages.csv`) and output preprocessed features to `data/preproceed/`.

### 2. Model Training
Edit `train.py` to set training parameters and model configuration.
Run:
```bash
python train.py
```
This will train an MLP model using the processed data and save the model to `model/models/`.

### 3. Prediction
Edit `predict.py` to set the input data and model path.
Run:
```bash
python predict.py
```
This will generate predictions and save results to `data/predict/`.

## Features
- Configurable preprocessing: AES encryption, filtering, keyword extraction
- Static and dynamic text embeddings: Word2Vec, TF-IDF, Qwen3, BGE-M3
- Deep learning model training: MLP with flexible architecture
- Easy-to-modify configuration via YAML and Python scripts
- Supports large-scale data and chunked processing

## Directory Structure
- `main.py`: Data preprocessing entry
- `train.py`: Model training entry
- `predict.py`: Prediction entry
- `src/`: Core modules (config, data processing, training, logging)
- `model/`: Model files and deep learning modules
- `data/`: Raw, processed, and prediction data
- `requirements.txt`: Python dependencies


## Experiment Records

SMS2Vec provides a three-layer DNN as a baseline model, which demonstrates strong performance on industrial datasets. Notably, the inclusion of SMS signature sequence features leads to significant improvements on the validation set.

| Method                                 | Val Acc | F1    | Recall | AUC    | KS     | Loss   |
|-----------------------------------------|---------|-------|--------|--------|--------|--------|
| Qwen3 Embedding + Word2Vec + MLP      | 0.7438  | 0.5196| 0.4243 | 0.8306 | 0.5120 | 0.5095 |
| Qwen3 Embedding + Word2Vec + Sign Seq + MLP| 0.8333  | 0.7011| 0.5989 | 0.9196 | 0.6543 | 0.2377 |

*Note: Experiment based on an small industrial dataset containing 89,248 message records from 31,296 users.*

*"Sign Seq" refers to signature ID sequence features.*


