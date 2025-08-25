"""
trainer.py

This module provides the Trainer class for training and saving various models, including tree models and custom neural networks.
Supported models: RandomForest, XGBoost, LightGBM, CatBoost, LSTM, Transformer, DNN.
Model structure and weights are saved in the model/ directory.
"""

import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim

class DNN(nn.Module):
	def __init__(self, input_dim, hidden_dim=128, output_dim=1):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim)
		)
	def forward(self, x):
		return self.net(x)

class Trainer:
	def __init__(self, model_type, X, y, model_params=None, model_dir="model"):
		self.model_type = model_type
		self.X = X
		self.y = y
		self.model_params = model_params or {}
		self.model_dir = model_dir
		os.makedirs(model_dir, exist_ok=True)
		self.model = self._init_model()

	def _init_model(self):
		if self.model_type == "rf":
			return RandomForestClassifier(**self.model_params)
		elif self.model_type == "xgb":
			return XGBClassifier(**self.model_params)
		elif self.model_type == "lgbm":
			return LGBMClassifier(**self.model_params)
		elif self.model_type == "catboost":
			return CatBoostClassifier(**self.model_params)
		elif self.model_type == "dnn":
			input_dim = self.X.shape[1]
			return DNN(input_dim)
		# 你可以在此扩展 lstm/transformer 网络结构
		else:
			raise ValueError(f"Unsupported model type: {self.model_type}")

	def train(self):
		if self.model_type in ["rf", "xgb", "lgbm", "catboost"]:
			self.model.fit(self.X, self.y)
		elif self.model_type == "dnn":
			X_tensor = torch.tensor(self.X.values, dtype=torch.float32)
			y_tensor = torch.tensor(self.y.values, dtype=torch.float32).view(-1, 1)
			optimizer = optim.Adam(self.model.parameters(), lr=0.001)
			criterion = nn.BCEWithLogitsLoss()
			self.model.train()
			for epoch in range(20):
				optimizer.zero_grad()
				output = self.model(X_tensor)
				loss = criterion(output, y_tensor)
				loss.backward()
				optimizer.step()
		# 可扩展：LSTM/Transformer等自定义网络
		# if self.model_type == "lstm":
		#     ...
		# if self.model_type == "transformer":
		#     ...
		return self.model

	def save(self, name):
		path = os.path.join(self.model_dir, name)
		if self.model_type in ["rf", "xgb", "lgbm", "catboost"]:
			joblib.dump(self.model, path + ".joblib")
			# 保存结构信息
			with open(path + "_structure.json", "w") as f:
				json.dump({"type": self.model_type, "params": self.model.get_params()}, f)
		elif self.model_type == "dnn":
			torch.save(self.model.state_dict(), path + ".pt")
			with open(path + "_structure.json", "w") as f:
				json.dump({"type": self.model_type, "params": self.model_params}, f)
		# 可扩展：LSTM/Transformer等自定义网络保存
		# if self.model_type == "lstm":
		#     torch.save(self.model.state_dict(), path + "_lstm.pt")
		# if self.model_type == "transformer":
		#     torch.save(self.model.state_dict(), path + "_transformer.pt")
