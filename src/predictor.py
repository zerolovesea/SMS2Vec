"""
predictor.py

This module provides the Predictor class for loading trained models and performing inference on new data.
Supports batch prediction and chunked prediction for large datasets.
"""

import joblib
import pandas as pd

class Predictor:
	def __init__(self, model_path: str):
		"""
		Load a trained model for inference.
		"""
		self.model = joblib.load(model_path)

	def predict(self, X):
		"""
		Predict labels for input features X.
		"""
		if hasattr(self.model, 'predict_proba'):
			return self.model.predict_proba(X)[:, 1]
		else:
			return self.model.predict(X)

	def predict_file(self, data_path, output_path=None, chunk_size=100000):
		"""
		Predict for a large CSV file, optionally in chunks.
		"""
		reader = pd.read_csv(data_path, chunksize=chunk_size)
		results = []
		for chunk in reader:
			X = chunk.drop(['phone', 'label'], axis=1, errors='ignore')
			preds = self.predict(X)
			chunk_result = chunk[['phone']].copy()
			chunk_result['score'] = preds
			results.append(chunk_result)
		result_df = pd.concat(results, ignore_index=True)
		if output_path:
			result_df.to_csv(output_path, index=False)
		return result_df
