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
- Static and dynamic feature extraction (TF-IDF, Word2Vec, BERT, RoBERTa)
- Supports multiple model types: RF, XGBoost, LightGBM, CatBoost, DNN
- Unified logging and modular code structure
- Large file chunk prediction supported



# SMS2Vec

本项目用于短信数据的特征工程、模型训练与推理。

## 目录结构
- src/ 主要代码模块
- model/ 训练好的模型文件
- data/ 数据文件
- docs/ 项目文档

## 环境构建
推荐使用 [uv](https://github.com/astral-sh/uv) 管理 Python 依赖。

```bash
uv venv
uv pip install -r requirements.txt
```

## 快速开始
详见 docs/usage.md。
