# SMS2Vec Project

## Project Overview
SMS2Vec is a modular pipeline for financial/text mining tasks, supporting data preprocessing, feature engineering, model training, and inference. It is designed for large-scale SMS datasets, supporting static (TF-IDF, Word2Vec) and dynamic (BERT, RoBERTa) feature extraction, and multiple model types (RF, XGBoost, LightGBM, CatBoost, DNN, etc.).

## Example Data Format
The input CSV should contain the following columns:

| id         | message                                                                                                                        | sign           | datetime | label |
|------------------|-------------------------------------------------------------------------------------------------------------------------------|----------------|----------|-------|
| ++/2wZ+SulUjk7E0VUZRjg== | [AutoBrand] Holiday Sale! Visit our showroom for exclusive test drive gifts. Buy now and get a $500 voucher. To unsubscribe reply STOP. | [AutoBrand] | 05:30.9  | 1     |

## Example Usage

### Data Preprocessing
```bash
python main_pipeline.py --mode preprocess --data data/input.csv --output data/preprocess/demo_preprocessed.csv --tag demo --static_vec tf-idf --dynamic_vec bert --index phone_id --is_train
```

### Model Training
```bash
python main_pipeline.py --mode train --data data/preprocess/demo_preprocessed.csv --model_type rf --tag demo
```

### Inference
```bash
python main_pipeline.py --mode predict --data data/predict.csv --model_path model/demo_rf.pkl --output data/predict_result.csv
```

## Features
- Data preprocessing with configurable stopwords and index column
- Static and dynamic feature extraction (TF-IDF, Word2Vec, RoBERTa, Qwen3, BGE-M3)
- ID Embedding
- Unified logging and modular code structure
- Large file chunk prediction supported


