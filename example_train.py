import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
from model.dl_modules.layers.mlp import MLP
from src.evaluation import evaluate


if __name__ == "__main__": 
    data_path = "data/preproceed/demo_train_qwen3_w2v_mean.csv"
    df = pd.read_csv(data_path)
    model_tag = 'MLP_Qwen3_W2V_mean'


    label_col = 'label'  
    X = df.drop(columns=[label_col,'id']).values
    y = df[label_col].values.astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    device = torch.device("mps")
    model = MLP(input_dim=X.shape[1], output_dim=2, hidden_dims=[128, 64, 32], activation='relu', dropout=0.1,
                dice_dim=2)
    model.to(device)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    epochs = 50
    best_acc = 0.0
    best_auc = 0

    best_model_state = None
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        acc, f1, recall, auc, ks = evaluate(model, val_loader, device)
        scheduler.step(acc)
        print(
            f"Epoch {epoch + 1}/{epochs} | Val Acc: {acc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f} | KS: {ks:.4f} | Loss: {loss:.4f}")

        if auc > best_auc:
            best_auc = auc

        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= 10:
            print(f"Validation accuracy has not improved for 10 consecutive epochs. Early stopping. Best validation accuracy: {best_acc:.4f}")
            break

    save_path = f"model/models/{model_tag}_{best_auc:.3f}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if best_model_state is not None:
        torch.save(best_model_state, save_path)
        print(f"Best model saved to {save_path}")
