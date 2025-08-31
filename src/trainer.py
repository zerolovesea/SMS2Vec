import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import evaluate
from model.dl_modules.dnn import DNN
from src.logger_manager import LoggerManager

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.model_tag = config.get('model_tag', 'MLP_BGE_M3_W2V_mean')
        self.data_path = config['data_path']
        self.label_col = config.get('label_col', 'label')
        self.batch_size = config.get('batch_size', 64)
        self.epochs = config.get('epochs', 50)
        self.patience = config.get('patience', 10)
        self.model_params = config.get('model_params', {})
        self.lr = config.get('lr', 1e-3)

        self._prepare_data()
        self._build_model()

        self.logger = LoggerManager.get_logger()

    def _prepare_data(self):
        df = pd.read_csv(self.data_path)
        sign_cols = [c for c in df.columns if c.startswith('sign_id_seq_')]
        feature_cols = [c for c in df.columns if c not in sign_cols + [self.label_col, 'id']]
        X = df[feature_cols].values.astype(float)
        
        sign_id_seq = df[sign_cols].fillna(0).values.astype(int) if sign_cols else None
        y = df[self.label_col].values.astype(int)

        X_train, X_val, sign_train, sign_val, y_train, y_val = train_test_split(
            X, sign_id_seq, y, test_size=0.2, random_state=42)
        if self.model_params.get('use_sign_embedding'):
            self.train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(sign_train, dtype=torch.long),
                torch.tensor(y_train, dtype=torch.long))
            self.val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(sign_val, dtype=torch.long),
                torch.tensor(y_val, dtype=torch.long))
        else:
            self.train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            self.val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        self.input_dim = X.shape[1]
        self.sign_seq_len = len(sign_cols) if sign_cols else None
        

    def _build_model(self):
        self.model = DNN(input_dim=self.input_dim, output_dim=2, **self.model_params)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

    def train(self):
        best_acc = 0.0
        best_auc = 0.0
        best_model_state = None
        no_improve_count = 0
        for epoch in range(self.epochs):
            self.model.train()
            for batch in self.train_loader:
                if self.model.use_sign_embedding:
                    x_batch, sign_id_seq, y_batch = batch
                    x_batch = x_batch.to(self.device)
                    sign_id_seq = sign_id_seq.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = self.model(x_batch, sign_id_seq)
                else:
                    x_batch, y_batch = batch
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = self.model(x_batch)
                self.optimizer.zero_grad()
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
            acc, f1, recall, auc, ks = evaluate(self.model, self.val_loader, self.device)
            self.scheduler.step(acc)
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs} | Val Acc: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f} | KS: {ks:.4f} | Loss: {loss:.4f}")
            if auc > best_auc:
                best_auc = auc
            if acc > best_acc:
                best_acc = acc
                best_model_state = self.model.state_dict()
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= self.patience:
                self.logger.info(f"Validation accuracy has not improved for {self.patience} consecutive epochs. Early stopping. Best validation accuracy: {best_acc:.4f}")
                break
        self._save_model(best_model_state, best_auc)

    def _save_model(self, state, best_auc):
        save_path = f"model/models/{self.model_tag}_{best_auc:.3f}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if state is not None:
            torch.save(state, save_path)
            self.logger.info(f"Best model saved to {save_path}")


