"""
data_processor.py

This module provides the DataProcessor class for SMS feature engineering, data cleaning, and preprocessing.
It supports static and dynamic feature extraction, including TF-IDF, Word2Vec, and BERT embeddings.
Logger is managed via LoggerManager. All stopwords are configurable.
"""
import os
import gc
import sys
import torch
import joblib
import jieba
import string

import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch import Tensor
import torch.nn.functional as F

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from modelscope import snapshot_download
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
from src.tool import decrypt_data
from src.logger_manager import LoggerManager


class DataProcessor:
    def __init__(self, 
                 use_aes: bool = False,
                 aes_key: bytes | None = None,
                 aes_iv: bytes | None = None,
                 additional_stopwords: set[str] | None = None,
                 filter_messages_setting: dict | None = None,
                 static_vec: str | None = None,
                 dynamic_vec: str | None = None,
                 w2v_config: dict = {},
                 tf_idf_config: dict = {},
                 dynamic_vec_pooling_method: str = 'mean',
                 language: str = 'cn'):

        self.logger = LoggerManager.get_logger()

        self.root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_aes = use_aes
        self.aes_key = aes_key
        self.aes_iv = aes_iv

        if language == 'cn':
            self.logger.info('Setting Chinese stopwords')
            self.stop_words_path = f'{self.root_path}/data/resources/cn_stopwords.txt'
        else:
            self.logger.info('Setting English stopwords')
            self.stop_words_path = f'{self.root_path}/data/resources/en_stopwords.txt'
        with open(self.stop_words_path, encoding='utf-8') as f:
            self.stopwords = set(line.strip() for line in f)

        if additional_stopwords:
            self.logger.info(f'Adding {len(additional_stopwords)} additional stopwords')
            self.stopwords = self.stopwords | additional_stopwords

        self.filter_messages_setting = filter_messages_setting
        self.static_vec = static_vec
        self.dynamic_vec = dynamic_vec
        self.w2v_config = w2v_config
        self.tf_idf_config = tf_idf_config

        self.static_vec_model_path = f'{self.root_path}/model/static_vec'
        os.makedirs(self.static_vec_model_path, exist_ok=True)
        self.dynamic_vec_pooling_method = dynamic_vec_pooling_method
        self.language = language

    def filter_messages(self, 
                        data: pd.DataFrame, 
                        filter_messages_setting: dict,
                        keep_verification: bool = True) -> pd.DataFrame:
        """
        Filters messages in a pandas DataFrame based on specified keywords.

        This method removes rows containing any of the `keywords_to_remove` in the 'message' column,
        except for messages containing the Chinese word '验证码' or the English word 'verification code'.
        These messages are kept only if they also contain any of the `keep_keywords`.

        Args:
            data (pd.DataFrame): Input DataFrame containing a 'message' column.
            filter_messages_setting (dict): A dictionary containing the filter settings.
                keywords_to_remove (list[str]): List of keywords to filter out from messages.
                keep_keywords (list[str]): List of keywords; messages containing '验证码' or 'verification code' are kept only if they also contain any of these.
            keep_verification (bool): Whether to keep messages containing '验证码' or 'verification code'.

        Returns:
            pd.DataFrame: Filtered DataFrame containing messages that do not match the removal criteria.

        Raises:
            Logs errors and returns an empty DataFrame if input is invalid or an exception occurs.

        Note:
            - Messages containing any keyword in `keywords_to_remove` are removed.
            - Messages containing '验证码' or 'verification code' are kept only if they also contain any keyword in `keep_keywords`.
            - If keep_verification is False, all messages containing '验证码'或'verification code'都被移除。
            - If `keywords_to_remove` or `keep_keywords` are empty, their respective filters are skipped.
        """
        try:
            if not isinstance(data, pd.DataFrame):
                self.logger.error("Input data must be a standard pandas DataFrame.")
                return pd.DataFrame()
            if 'message' not in data.columns:
                self.logger.error("Input DataFrame must contain 'message' column, which stands for the sms text content.")
                return pd.DataFrame()
            self.logger.info(f"Original data size: {len(data)}")

            keep_pattern = ''
            if filter_messages_setting.get("keep_keywords"):
                keep_pattern = '|'.join([str(k) for k in filter_messages_setting["keep_keywords"] if k])
            if keep_pattern:
                keep_mask = data['message'].str.contains(keep_pattern, case=False, na=False)
            else:
                keep_mask = pd.Series([False]*len(data), index=data.index)

            if filter_messages_setting.get("keywords_to_remove"):
                remove_pattern = '|'.join([str(k) for k in filter_messages_setting["keywords_to_remove"] if k])
            else:
                remove_pattern = ''
            if remove_pattern:
                remove_mask = data['message'].str.contains(remove_pattern, case=False, na=False)
            else:
                remove_mask = pd.Series([False]*len(data), index=data.index)
            has_verification_cn = data['message'].str.contains('验证码', case=False, na=False)
            has_verification_en = data['message'].str.contains('verification code', case=False, na=False)
            has_verification = has_verification_cn | has_verification_en
            
            self.logger.info(f"SMS filter configuration: keywords_to_remove={filter_messages_setting.get('keywords_to_remove')}, keep_keywords={filter_messages_setting.get('keep_keywords')}, keep_verification={keep_verification}")

            df_keep = data[keep_mask]
            if keep_verification:
                other_mask = (~keep_mask) & (~remove_mask) & (~has_verification | (has_verification & keep_mask))
            else:
                other_mask = (~keep_mask) & (~remove_mask) & (~has_verification == False)
            df_other = data[other_mask]
            df_filtered = pd.concat([df_keep, df_other]).drop_duplicates().reset_index(drop=True)

            self.logger.info(f"Total messages kept after filtering: {len(df_filtered)}")
            return df_filtered
        except Exception as e:
            self.logger.error(f"Error in filter_messages: {e}")
            return pd.DataFrame()

    def tokenizer(self, text: str) -> list[str]:
        """
        Tokenize text according to self.language ('cn' for Chinese, 'en' for English).
        For English, uses nltk.word_tokenize if available, otherwise str.split().
        For Chinese, uses jieba.lcut.
        """
        if self.language == 'cn':
            words = jieba.lcut(text)
        else:
            words = word_tokenize(text)

        return [w for w in words if w.strip() and w not in self.stopwords and not self.is_punctuation(w)]

    def is_punctuation(self, word: str) -> bool:
        chinese_punctuation = '，。！？；：""''（）【】《》、·…—'
        english_punctuation = string.punctuation
        if word.isdigit() or word.replace('.', '').isdigit():
            return True
        return all(char in chinese_punctuation + english_punctuation for char in word)


    def create_dynamic_vec(self, dynamic_vec, texts, device, batch_size=32, max_length=128, language='cn'):
        if dynamic_vec == 'bge-m3':
            model_dir = snapshot_download('BAAI/bge-m3', cache_dir='./model/dynamic_vec')
            model = BGEM3FlagModel(model_dir, use_fp16=True)
            return model.encode(texts, batch_size=batch_size)['dense_vecs']

        elif dynamic_vec == 'qwen3':
            model_dir = snapshot_download('Qwen/Qwen3-Embedding-0.6B', cache_dir='./model/dynamic_vec')
            model = SentenceTransformer(model_dir)
            
            all_embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Qwen3 Embedding", leave=True, dynamic_ncols=True):
                batch_embeddings = model.encode(texts[i:i+batch_size])
                all_embeddings.extend(batch_embeddings)
                del batch_embeddings
                gc.collect()
                torch.mps.empty_cache()
            return all_embeddings

        elif dynamic_vec in ['bert', 'roberta']:
            if language == 'en':
                model_name = 'bert-base-uncased' if dynamic_vec == 'bert' else 'roberta-base'
            else:
                model_name = 'google-bert/bert-base-chinese' if dynamic_vec == 'bert' else 'hfl/chinese-roberta-wwm-ext'
            tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir='./model/dynamic_vec')
            model = BertModel.from_pretrained(model_name, cache_dir='./model/dynamic_vec')

            # embedding_model_path = '/home/zhoufeng/.cache/modelscope/hub/models/dienstag/chinese-roberta-wwm-ext'
            # tokenizer = BertTokenizer.from_pretrained(embedding_model_path)
            # model = BertModel.from_pretrained(embedding_model_path)

            model.eval()
            model.to(device)
            all_embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc=f"{dynamic_vec} Embedding", leave=True, dynamic_ncols=True):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.extend(batch_embeddings)
            return all_embeddings

        else:
            raise ValueError(f"Unsupported dynamic_vec type: {dynamic_vec}")

    def embedding_pooling(self, data: pd.DataFrame, pooling: str = 'mean') -> pd.DataFrame:
        """
        Pooling embeddings for each user using the specified pooling method.

        Args:
            data (pd.DataFrame): DataFrame containing 'id' and embedding columns.
            pooling (str): Pooling method, one of ['mean', 'max', 'min', 'sum'].

        Returns:
            pd.DataFrame: DataFrame with pooled embeddings per user.
        """
        embedding_columns = [col for col in data.columns if col.startswith('embedding_')]
        if not embedding_columns:
            self.logger.warning("Cannot find any embedding columns in DataFrame.")
            return data[['id']].drop_duplicates()
        if pooling == 'mean':
            result = data.groupby('id')[embedding_columns].mean().reset_index()
        elif pooling == 'max':
            result = data.groupby('id')[embedding_columns].max().reset_index()
        elif pooling == 'min':
            result = data.groupby('id')[embedding_columns].min().reset_index()
        elif pooling == 'sum':
            result = data.groupby('id')[embedding_columns].sum().reset_index()
        elif pooling == 'attention':
            from model.dl_modules.attention_pooling import AttentionPooling
            embedding_dim = len(embedding_columns)
            attn_pool = AttentionPooling(embedding_dim).to(self.device)
            pooled_list = []
            for uid, group in data.groupby('id'):
                emb = torch.tensor(group[embedding_columns].values, dtype=torch.float32, device=self.device)
                pooled = attn_pool(emb).detach().cpu().numpy()
                pooled_list.append({'id': uid, **{f'embedding_{i}': pooled[i] for i in range(embedding_dim)}})
            result = pd.DataFrame(pooled_list)
        else:
            self.logger.warning(f"Unknown pooling method '{pooling}', defaulting to mean.")
            result = data.groupby('id')[embedding_columns].mean().reset_index()
        self.logger.info(f'Embedding pooling ({pooling}) completed, unique user count: {len(result)}, feature dimension: {len(embedding_columns)}')
        return result  
    
    def create_time_features(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_work_time'] = df['hour'].between(9, 17).astype(int)
        df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 22 or x < 6 else 0)
        user_time_features = df.groupby('id').agg({
            'hour': ['mean', 'std', 'nunique'],
            'day_of_week': ['nunique'],
            'is_weekend': ['mean', 'sum'],
            'is_work_time': ['mean', 'sum'],
            'is_night': ['mean', 'sum']
        }).reset_index()
        user_time_features.columns = ['id'] + ['_'.join(col) for col in user_time_features.columns[1:]]
        return user_time_features

    def create_sign_diversity_features(self, df):
        sign_stats = df.groupby('id').agg({
            'sign': ['nunique', 'count'],
        }).reset_index()
        sign_stats.columns = ['id', 'unique_signs', 'total_messages']
        sign_stats['sign_diversity_ratio'] = sign_stats['unique_signs'] / sign_stats['total_messages']
        most_common_sign_ratio = df.groupby('id')['sign'].apply(
            lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 and len(x.value_counts()) > 0 else 0
        ).reset_index()
        most_common_sign_ratio.columns = ['id', 'dominant_sign_ratio']
        return sign_stats.merge(most_common_sign_ratio, on='id')

    def create_message_content_features(self, df):
        df['message_length'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()
        df['digit_count'] = df['message'].str.count(r'\d')
        df['punctuation_count'] = df['message'].str.count(r'[，。！？；：""''（）【】《》、·…—]')
        df['amount_count'] = df['message'].str.count(r'\d+元|\d+\.?\d*万|\d+\.?\d*千')
        loan_keywords = ['贷款', '借钱', '放款', '额度', '利息', '还款', '分期', '信贷']
        risk_keywords = ['逾期', '催收', '欠款', '违约', '法律', '起诉', '征信']
        promo_keywords = ['优惠', '活动', '折扣', '免费', '赠送', '抽奖', '中奖']
        for keyword_list, prefix in [(loan_keywords, 'loan'), (risk_keywords, 'risk'), (promo_keywords, 'promo')]:
            df[f'{prefix}_keyword_count'] = df['message'].apply(
                lambda x: sum(1 for kw in keyword_list if kw in x)
            )
        content_features = df.groupby('id').agg({
            'message_length': ['mean', 'std', 'min', 'max'],
            'word_count': ['mean', 'std'],
            'digit_count': ['mean', 'sum'],
            'punctuation_count': ['mean', 'sum'], 
            'amount_count': ['sum'],
            'loan_keyword_count': ['sum', 'mean'],
            'risk_keyword_count': ['sum', 'mean'],
            'promo_keyword_count': ['sum', 'mean']
        }).reset_index()
        content_features.columns = ['id'] + ['_'.join(col) for col in content_features.columns[1:]]
        return content_features

    def create_area_features(self, df):
        df['area_code'] = df['id'].str[3:7] 
        df['carrier_code'] = df['id'].astype(str).str[:3]
        carrier_stats = df.groupby('carrier_code').agg({
            'id': 'nunique',
            'message': 'count'
        }).reset_index()
        carrier_stats.columns = ['carrier_code', 'carrier_user_count', 'carrier_message_count']
        df = df.merge(carrier_stats, on='carrier_code', how='left')
        return df

    def create_messages_distribution_features(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df['contain_slash'] = df['message'].str.count('/')
        df['contain_url'] = df['message'].str.count('://')
        user_features = df.groupby('id').agg({
            'message': ['count'],
            'sign': ['nunique'],
            'datetime': ['min', 'max'],
            'contain_slash': ['sum'],
            'contain_url': ['sum']
        }).reset_index()
        user_features.columns = ['id', 'message_count', 'sign_unique_count', 'datetime_min', 'datetime_max', 'slash_count', 'url_count']
        user_features['time_span_days'] = (user_features['datetime_max'] - user_features['datetime_min']).dt.total_seconds() / (24 * 3600)
        user_features['time_span_days'] = user_features['time_span_days'].replace(0, 1/24)
        user_features['message_frequency'] = user_features['message_count'] / user_features['time_span_days']
        today = datetime.datetime.now()
        user_features['days_since_last_message'] = (today - user_features['datetime_max']).dt.total_seconds() / (24 * 3600)
        user_features['days_since_last_message'] = user_features['days_since_last_message'].replace(0, 1/24)
        user_features = user_features.drop(['datetime_min', 'datetime_max', 'time_span_days'], axis=1)
        return user_features

    def create_comprehensive_features(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')
        df = df.dropna(subset=['datetime'])

        time_features = self.create_time_features(df)
        content_features = self.create_message_content_features(df)
        sign_features = self.create_sign_diversity_features(df)
        df_with_area = self.create_area_features(df)
        area_features = df_with_area.groupby('id').agg({
            'carrier_user_count': 'first',
            'carrier_message_count': 'first'
        }).reset_index()
        messages_distribution_features = self.create_messages_distribution_features(df)
        all_features = time_features
        feature_dfs = [content_features,sign_features,area_features,messages_distribution_features]
        for i, feature_df in enumerate(feature_dfs):
            all_features = all_features.merge(feature_df, on='id', how='left')
        self.logger.info(f"Comprehensive features shape: {all_features.shape}")
        for col in all_features.columns:
            if all_features[col].isnull().sum()>0:
                all_features[col] = all_features[col].fillna(0)
        return all_features

    def preprocess_batch(self, 
                         data: pd.DataFrame, 
                         is_training: bool=True, 
                         data_tag: str = 'demo',
                         chunk_id: int|None=None):
        df_label = None
        if is_training:
            df_label = data[['id', 'label']]

        if self.filter_messages_setting:
            data = self.filter_messages(data=data, filter_messages_setting=self.filter_messages_setting)

        # create static vector representations
        data_static_vec = data.copy()
        data_static_vec['text'] = data_static_vec['message']
        data_merged_static_vec = data_static_vec.groupby('id').agg({
            'text': lambda x: ' '.join(x)
        }).reset_index()
    
        if is_training:
            if self.static_vec == 'tf-idf':
                self.logger.info(f'Using TF-IDF vectorizer')
                vectorizer = TfidfVectorizer(
                    tokenizer=self.tokenizer, **self.tf_idf_config
                )
                static_matrix = vectorizer.fit_transform(data_merged_static_vec['text'])
                try:
                    static_matrix = static_matrix.toarray()
                except Exception:
                    static_matrix = np.array(static_matrix)

                joblib.dump(vectorizer, f'{self.static_vec_model_path}/{self.static_vec}_model.pkl')
                self.logger.info(f'Static vectorizer saved: {self.static_vec_model_path}/{self.static_vec}_model.pkl')

            elif self.static_vec == 'word2vec':
                self.logger.info(f'Using Word2Vec vectorizer')
                all_messages = data_static_vec['message'].tolist()
                sentences = [jieba.lcut(msg) for msg in all_messages if msg.strip()]

                if self.w2v_config:
                    self.logger.info(f'Using Word2Vec with custom config')
                    vectorizer = Word2Vec(
                        sentences=sentences,
                        vector_size=self.w2v_config.get('vector_size', 300),
                        window=self.w2v_config.get('window', 5),
                        min_count=self.w2v_config.get('min_count', 2),
                        workers=self.w2v_config.get('workers', 4),
                        epochs=self.w2v_config.get('epochs', 10),
                        sg=self.w2v_config.get('sg', 1)
                    )
                else:
                    self.logger.info(f'Using Word2Vec with default config')
                    vectorizer = Word2Vec(
                        sentences=sentences,
                        vector_size=300,
                        window=5,
                        min_count=2,
                        workers=4,
                        epochs=10,
                        sg=1
                    )
                user_vectors = []
                for id in tqdm(data_merged_static_vec['id'], desc="Fitting Word2Vec user vector"):
                    user_text = data_merged_static_vec[data_merged_static_vec['id'] == id]['text'].iloc[0]
                    user_words = jieba.lcut(user_text)
                    word_vectors = [vectorizer.wv[word] for word in user_words if word in vectorizer.wv]
                    if word_vectors:
                        user_vector = np.mean(word_vectors, axis=0)
                    else:
                        user_vector = np.zeros(300)
                    user_vectors.append(user_vector)
                static_matrix = np.array(user_vectors)

                vectorizer.save(f'{self.static_vec_model_path}/{self.static_vec}_model.bin')
                self.logger.info(f'Word2Vec model saved: {self.static_vec_model_path}/{self.static_vec}_model.bin')

        else:
            if self.static_vec == 'tf-idf':
                model_file = f'{self.static_vec_model_path}/{self.static_vec}_model.pkl'
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"TF-IDF model file not found: {model_file}")
                vectorizer = joblib.load(model_file)
                self.logger.info(f'Static vectorizer loaded: {model_file}')
                static_matrix = vectorizer.transform(data_merged_static_vec['text'])
                if hasattr(static_matrix, 'toarray'):
                    static_matrix = static_matrix.toarray()
            elif self.static_vec == 'word2vec':
                model_file = f'{self.static_vec_model_path}/{self.static_vec}_model.bin'
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Word2Vec model file not found: {model_file}")
                vectorizer = Word2Vec.load(model_file)
                self.logger.info(f'Word2Vec model loaded: {model_file}')
                user_vectors = []
                for id in tqdm(data_merged_static_vec['id'], desc="Fitting Word2Vec user vector"):
                    user_text = data_merged_static_vec[data_merged_static_vec['id'] == id]['text'].iloc[0]
                    user_words = self.tokenizer(user_text)
                    word_vectors = [vectorizer.wv[word] for word in user_words if word in vectorizer.wv]
                    if word_vectors:
                        user_vector = np.mean(word_vectors, axis=0)
                    else:
                        user_vector = np.zeros(300)
                    user_vectors.append(user_vector)
                static_matrix = np.array(user_vectors)

        if not isinstance(static_matrix, np.ndarray):
            try:
                static_matrix = static_matrix.toarray()
            except Exception:
                static_matrix = np.array(static_matrix)

        data_static_vec = pd.DataFrame(static_matrix, columns=[f'static_vec_{i}' for i in range(static_matrix.shape[1])])
        data_static_vec['id'] = data_merged_static_vec['id']
        del data_merged_static_vec, static_matrix
        gc.collect()

        # Create dynamic vector representations
        data_dynamic_vec = data.copy()
        dynamic_embeddings = self.create_dynamic_vec(
            dynamic_vec=self.dynamic_vec,
            device=self.device,
            texts=data_dynamic_vec['message'].tolist(),
        )
        dynamic_embeddings_array = np.array(dynamic_embeddings)
        dynamic_embedding_dim = dynamic_embeddings_array.shape[1]
        self.logger.info(f'Dynamic Embedding dim: {dynamic_embedding_dim}')
        df_embedding = pd.DataFrame(dynamic_embeddings_array, columns=[f'embedding_{i}' for i in range(dynamic_embedding_dim)])
        data_dynamic_vec = pd.concat([data_dynamic_vec.reset_index(drop=True), df_embedding], axis=1)
        embedding_columns = [col for col in data_dynamic_vec.columns if col.startswith('embedding_')]
        data_for_pooling = data_dynamic_vec[['id'] + embedding_columns].copy()
        data_dynamic_vec = self.embedding_pooling(data_for_pooling, pooling=self.dynamic_vec_pooling_method)

        del dynamic_embeddings, dynamic_embeddings_array, df_embedding, data_for_pooling
        gc.collect()

        # create extra features
        df_features_extra = self.create_comprehensive_features(data)

        for data in [data_static_vec, data_dynamic_vec, df_features_extra]:
            data['id'] = data['id'].astype(str)

        df_features = pd.merge(data_dynamic_vec, data_static_vec, on='id', how='inner')
        df_features = pd.merge(df_features, df_features_extra, on='id', how='left')

        del data_dynamic_vec, df_features_extra, data_static_vec
        gc.collect()

        # merge labels
        if is_training and df_label is not None:
            df_features = df_features.merge(
                df_label,
                on='id',
                how='left'
            )
            
        if chunk_id is not None:
            output_path = f'{self.root_path}/data/preproceed/{data_tag}_chunk_{chunk_id}.csv'
        else:
            output_path = f'{self.root_path}/data/preproceed/{data_tag}.csv'

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_features.to_csv(output_path, index=False)
        self.logger.info(f'Feature data saved: {output_path}, shape: {df_features.shape}')
        return output_path

    def preprocess(self, 
                   data_path: str, 
                   enc_col: str|None = None,
                   data_tag: str = '',
                   chunk_size: int = 200000):  
        self.logger.info(f'Starting preprocessing for data: {data_path}, data_tag: {data_tag}, chunk_size: {chunk_size}')
        output_file_path = f'{self.root_path}/data/preproceed/{data_tag}.csv'
        if os.path.exists(output_file_path):
            self.logger.info(f'Output file already exists: {output_file_path}, skip preprocessing.')
            return output_file_path

        is_training = False
        data = pd.read_csv(data_path, encoding='utf-8', on_bad_lines='skip')

        if self.use_aes and enc_col:
            self.logger.info(f'Using AES encryption for column: "{enc_col}", transforming to column: "id" ')
            data = decrypt_data(data=data, aes_key=self.aes_key, aes_iv=self.aes_iv, enc_col=enc_col)

        if 'label' in data.columns:
            is_training = True
            self.logger.info(f"Detected 'label' column, setting is_training=True")
        else:
            self.logger.info(f"No 'label' column detected, setting is_training=False")

        total_rows = len(data)
        if is_training:
            self.logger.info(f'Training mode: processing all data at once, total rows: {total_rows}')
            self.preprocess_batch(
                data=data,
                is_training=is_training,
                data_tag=data_tag
            )
        else:
            if total_rows > chunk_size:
                self.logger.info(f'Total rows: {total_rows}, preprocessing in chunks of {chunk_size} rows')
                chunk_files_path = []
                num_chunks = (total_rows + chunk_size - 1) // chunk_size
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, total_rows)
                    self.logger.info(f"Processing chunk {i+1}/{num_chunks}, row range: {start_idx}-{end_idx}")
                    data_chunk = data.iloc[start_idx:end_idx].copy()
                    chunk_file_path = self.preprocess_batch(
                        data=data_chunk,
                        is_training=is_training,
                        data_tag=data_tag,
                        chunk_id=i)
                    chunk_files_path.append(chunk_file_path)
                    del data_chunk
                    gc.collect()
                    self.logger.info(f"Chunk {i+1} processing completed")
                self.logger.info("Starting to write all chunk results to final file (append mode)")
                
                first = True
                for i, chunk_file_path in enumerate(chunk_files_path):
                    chunk_data = pd.read_csv(chunk_file_path)
                    chunk_data.to_csv(output_file_path, mode='w' if first else 'a', header=first, index=False)
                    self.logger.info(f'Chunk {i+1} appended to {output_file_path}')
                    first = False
                    del chunk_data
                    gc.collect()
                self.logger.info(f'All chunks written to {output_file_path}')
            else:
                self.preprocess_batch(
                    data=data,
                    is_training=is_training,
                    data_tag=data_tag
                )

            
