
# SMS2Vec

## Project Overview
SMS2Vec is a modular pipeline for SMS text mining and classification. The project aims to process large-scale SMS datasets, extract static and dynamic text features, and train deep learning models for classification or prediction tasks. It supports flexible configuration, multiple feature extraction methods, and efficient training/inference workflows.

## Purpose
The goal of SMS2Vec is to provide a robust and extensible framework for:
- Preprocessing raw SMS data, including encryption/decryption, filtering, and keyword extraction
- Generating static (Word2Vec, TF-IDF) and dynamic (Qwen3, BGE-M3, RoBERTa) text embeddings
- Training deep learning models (MLP) for SMS classification
- Making predictions on new SMS data

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

## License
See LICENSE file for details.


