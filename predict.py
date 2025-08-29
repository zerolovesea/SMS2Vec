import os
import torch
import pandas as pd
from model.dl_modules.mlp import MLP

if __name__ == "__main__":
    data_path = "data/preproceed/demo_train_bge_m3_w2v_mean.csv"
    df = pd.read_csv(data_path)

    data_tag = ''

    X = df.drop(columns=['id']).values

    device = torch.device("mps")
    model = MLP(input_dim=X.shape[1], output_dim=2, hidden_dims=[128, 64, 32], activation='relu', dropout=0.1, dice_dim=2)
    model.to(device)

    model_path = "model/models/MLP_BGE_M3_W2V_mean_best.pt"  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()

    df['score'] = probs[:, 1].cpu().numpy()
    df.to_csv(f"data/predict/{data_tag}.csv", index=False)
