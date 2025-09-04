import os
import torch
import argparse
import pandas as pd
from model.dl_modules.dnn import DNN
from src.logger_manager import LoggerManager
from torch.utils.data import DataLoader, TensorDataset

def predict(model_path, data_path, data_key, batch_size=10000, device='cpu'):
    logger = LoggerManager.get_logger()
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    output_path = f"data/predict/{data_key}.csv"


    logger.info(f"Reading data from {data_path} in chunks of {batch_size}")
    model_params = checkpoint.get('model_params', None)
    input_dim = checkpoint.get('input_dim', None)
    model = DNN(input_dim=input_dim, output_dim=2, **model_params)
    model.load_state_dict(checkpoint if isinstance(checkpoint, dict) else checkpoint)
    model.to(device)
    model.eval()
    logger.info("Model loaded and ready for prediction.")

    results = []
    chunk_iter = pd.read_csv(data_path, chunksize=batch_size)
    chunk_id = 0

    
    for chunk in chunk_iter:
        chunk_id += 1
        sign_cols = [c for c in chunk.columns if c.startswith('sign_id_seq_')]
        feature_cols = [c for c in chunk.columns if c not in sign_cols + ['id']]
        X = chunk[feature_cols].values.astype(float)
        sign_id_seq = chunk[sign_cols].fillna(0).values.astype(int) if sign_cols else None
        ids = chunk['id'].astype(str).tolist()
        input_dim = X.shape[1] if input_dim is None else input_dim
        logger.info(f"Processing chunk {chunk_id} with {len(ids)} rows, input_dim={input_dim}")
        if sign_id_seq is not None:
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(sign_id_seq, dtype=torch.long),
                torch.tensor(range(len(ids)), dtype=torch.long)
            )
        else:
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(range(len(ids)), dtype=torch.long)
            )
        loader = DataLoader(dataset, batch_size=batch_size)
        for batch in loader:
            if sign_id_seq is not None:
                x_batch, sign_batch, idx_batch = batch
                x_batch = x_batch.to(device)
                sign_batch = sign_batch.to(device)
                logits = model(x_batch, sign_batch)
            else:
                x_batch, idx_batch = batch
                x_batch = x_batch.to(device)
                logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            batch_ids = [ids[i.item()] for i in idx_batch]
            df_pred = pd.DataFrame({'id': batch_ids, 'score': probs})
            results.append(df_pred)

    df_all = pd.concat(results, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_csv(output_path, index=False)
    logger.info(f"Prediction results saved to {output_path}, total {len(df_all)} rows.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to preprocessed csv file')
    parser.add_argument('--data_key', type=str, default='demo_predict_result', help='Predict result key')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    predict(args.model, args.data, args.data_key, args.batch_size, args.device)

    # python predict.py --model /home/zhoufeng/sms2vec/model/models/DNN_QWEN3_W2V_mean_HFQ_ABSEED_250820_sign_sequences_0.897.pt --data /home/zhoufeng/sms2vec/data/preproceed/hfq_stacking_apply_qwen3_w2v_mean_sign_sequences.csv --data_key TMP_DNN-QWEN3-W2V-SEQ-HFQ-ABSEED-250820-20250904 --batch_size 10000 --device cuda