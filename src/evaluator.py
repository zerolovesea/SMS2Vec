"""
evaluator.py

Evaluation utilities for model training and prediction in SMS2Vec project.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

class Evaluator:
    """
    Provides static methods for evaluating classification models.
    """
    @staticmethod
    def evaluate(y_true, y_pred, y_pred_proba=None, average=None):
        metrics = {}
        # 自动判断分类类型
        unique_labels = np.unique(y_true)
        if average is None:
            average = 'binary' if len(unique_labels) == 2 else 'macro'
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average)
        metrics['recall'] = recall_score(y_true, y_pred, average=average)
        metrics['f1'] = f1_score(y_true, y_pred, average=average)
        if y_pred_proba is not None:
            # 多分类roc_auc需指定multi_class
            if len(unique_labels) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        return metrics

    @staticmethod
    def print_metrics(metrics):
        print("Model Evaluation Results:")
        for k, v in metrics.items():
            if k == 'classification_report' or k == 'confusion_matrix':
                print(f"{k}:\n{v}")
            else:
                print(f"{k}: {v}")
