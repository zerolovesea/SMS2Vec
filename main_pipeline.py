"""
main_pipeline.py

A universal entry script for SMS2Vec project. Supports data preprocessing, model training, and inference via argparse.
"""
import argparse
import pandas as pd
import os
from src.data_processor import DataProcessor
from src.trainer import Trainer
from src.predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description='SMS2Vec Pipeline')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'predict'], required=True, help='Task mode')
    parser.add_argument('--data', type=str, help='Input data path')
    parser.add_argument('--output', type=str, help='Output path (for preprocess/predict)')
    parser.add_argument('--model_type', type=str, default='rf', help='Model type: rf/xgb/lgbm/catboost/dnn')
    parser.add_argument('--model_path', type=str, help='Model path (for predict)')
    parser.add_argument('--tag', type=str, default='demo', help='Tag for output files')
    parser.add_argument('--static_vec', type=str, default='tf-idf', choices=['tf-idf', 'word2vec'], help='Static vector type')
    parser.add_argument('--dynamic_vec', type=str, default=None, choices=['bert', 'roberta'], help='Dynamic vector type (bert/roberta)')
    parser.add_argument('--is_train', action='store_true', help='Is training mode (for preprocess)')
    parser.add_argument('--chunk_size', type=int, default=100000, help='Chunk size for large files')
    parser.add_argument('--index', type=str, default='phone', help='Index column name (default: phone), standing for unique user identifier')
    args = parser.parse_args()

    if args.mode == 'preprocess':
        try:
            df = pd.read_csv(args.data)
            assert args.index in df.columns, f"Error: Index column '{args.index}' not found in input data! Columns: {list(df.columns)}"
            if 'label' in df.columns:
                df_label = df[[args.index, 'label']]
            else:
                df_label = None
            processor = DataProcessor()
            output_path = args.output or f"data/preprocess/{args.tag}_preprocessed.csv"
            result_path = processor.preprocess(
                df,
                data_tag=args.tag,
                static_vec=args.static_vec,
                dynamic_vec=args.dynamic_vec,
                is_training=args.is_train,
                df_label=df_label
            )
            if result_path != output_path:
                os.rename(result_path, output_path)
            print(f"Preprocessed data saved to: {output_path}")
        except AssertionError as e:
            print(str(e))
        except Exception as e:
            print(f"Preprocess failed: {e}")

    elif args.mode == 'train':
        df = pd.read_csv(args.data)
        X = df.drop(['phone', 'label'], axis=1)
        y = df['label']
        trainer = Trainer(model_type=args.model_type, X=X, y=y)
        model = trainer.train()
        model_name = f"{args.tag}_{args.model_type}"
        trainer.save(model_name)
        print(f"Model saved to: model/{model_name}.pkl or .pt")

    elif args.mode == 'predict':
        predictor = Predictor(args.model_path)
        result_df = predictor.predict_file(args.data, output_path=args.output, chunk_size=args.chunk_size)
        print(f"Prediction results saved to: {args.output}")

if __name__ == "__main__":
    main()
