import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

def ks_score(y_true, y_pred_prob):
    from scipy.stats import ks_2samp
    return ks_2samp(y_pred_prob[y_true == 0], y_pred_prob[y_true == 1]).statistic

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            if model.use_sign_embedding:
                x_batch, sign_id_seq, y_batch = batch
                x_batch = x_batch.to(device)
                sign_id_seq = sign_id_seq.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch, sign_id_seq)
            else:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    ks = ks_score(np.array(all_labels), np.array(all_probs))
    return acc, f1, recall, auc, ks