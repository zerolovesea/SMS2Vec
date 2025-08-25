

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
import base64
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.logger_manager import LoggerManager
from sklearn.decomposition import TruncatedSVD
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


class DataProcessor:
    def __init__(self, 
                 use_aes: bool = False,
                 aes_key: bytes | None = None,
                 aes_iv: bytes | None = None,
                 set_cn_stopwords: bool = False, 
                 additional_stopwords: set[str] | None = None,
                 filter_messages_setting: dict | None = None,
                 static_vec: str | None = None,
                 dynamic_vec: str | None = None,
                 language: str = 'cn'):

        self.root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_aes = use_aes
        self.aes_key = aes_key
        self.aes_iv = aes_iv

        if set_cn_stopwords:
            self.stop_words_path = f'{self.root_path}/data/resources/cn_stopwords.txt'
        else:
            self.stop_words_path = f'{self.root_path}/data/resources/en_stopwords.txt'
        with open(self.stop_words_path, encoding='utf-8') as f:
            self.stopwords = set(line.strip() for line in f)

        if additional_stopwords:
            self.stopwords = self.stopwords | additional_stopwords

        self.filter_messages_setting = filter_messages_setting
        self.static_vec = static_vec
        self.dynamic_vec = dynamic_vec
        self.language = language

        self.logger = LoggerManager.get_logger()

    def aes_encrypt(self, text: str) -> str:
        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_iv)
        padded_text = pad(text.encode('utf-8'), AES.block_size)
        encrypted = cipher.encrypt(padded_text)
        return base64.b64encode(encrypted).decode('utf-8')

    def aes_decrypt(self, enc_text: str) -> str:
        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_iv)
        decrypted = cipher.decrypt(base64.b64decode(enc_text))
        unpadded = unpad(decrypted, AES.block_size)
        return unpadded.decode('utf-8')

    def decrypt_data(self, data: pd.DataFrame, enc_col: str = 'phone_id') -> pd.DataFrame:
        assert enc_col in data.columns, f"Encrypted column '{enc_col}' does not exist in DataFrame"
        data['id'] = data[enc_col].apply(self.aes_decrypt)
        data.drop(columns=[enc_col], inplace=True)
        return data

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
            
            self.logger.info(f"Filter configuration: keywords_to_remove={filter_messages_setting.get('keywords_to_remove')}, keep_keywords={filter_messages_setting.get('keep_keywords')}, keep_verification={keep_verification}")

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
            from nltk.tokenize import word_tokenize
            words = word_tokenize(text)

        return [w for w in words if w.strip() and w not in self.stopwords and not self.is_punctuation(w)]

    def is_punctuation(self, word: str) -> bool:
        chinese_punctuation = '，。！？；：""''（）【】《》、·…—'
        english_punctuation = string.punctuation
        if word.isdigit() or word.replace('.', '').isdigit():
            return True
        return all(char in chinese_punctuation + english_punctuation for char in word)

    def encode_texts_batch(self, texts, tokenizer, model, batch_size=32, is_training=True):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="batch encoding"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(cls_embeddings)
        return all_embeddings

    def embedding_pooling(self, df):
        embedding_columns = [col for col in df.columns if col.startswith('embedding_')]
        if not embedding_columns:
            self.logger.warning("没有找到embedding列")
            return df[['phone']].drop_duplicates()
        result = df.groupby('phone')[embedding_columns].mean().reset_index()
        self.logger.info(f'Embedding聚合完成，用户数: {len(result)}, 特征维度: {len(embedding_columns)}')
        return result

    # def concat_embedding_tfidf(self, embedding_df, tfidf_df):
    #     embedding_df['phone'] = embedding_df['phone'].astype(object)
    #     tfidf_df['phone'] = tfidf_df['phone'].astype(object)
    #     merged_df = embedding_df.merge(
    #             tfidf_df.drop(['area_code'], axis=1, errors='ignore'), 
    #             on=['phone'], 
    #             how='inner'
    #         )
    #     return merged_df

    def create_time_features(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_work_time'] = df['hour'].between(9, 17).astype(int)
        df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 22 or x < 6 else 0)
        user_time_features = df.groupby('phone').agg({
            'hour': ['mean', 'std', 'nunique'],
            'day_of_week': ['nunique'],
            'is_weekend': ['mean', 'sum'],
            'is_work_time': ['mean', 'sum'],
            'is_night': ['mean', 'sum']
        }).reset_index()
        user_time_features.columns = ['phone'] + ['_'.join(col) for col in user_time_features.columns[1:]]
        return user_time_features

    def create_sign_diversity_features(self, df):
        sign_stats = df.groupby('phone').agg({
            'sign': ['nunique', 'count'],
        }).reset_index()
        sign_stats.columns = ['phone', 'unique_signs', 'total_messages']
        sign_stats['sign_diversity_ratio'] = sign_stats['unique_signs'] / sign_stats['total_messages']
        most_common_sign_ratio = df.groupby('phone')['sign'].apply(
            lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 and len(x.value_counts()) > 0 else 0
        ).reset_index()
        most_common_sign_ratio.columns = ['phone', 'dominant_sign_ratio']
        return sign_stats.merge(most_common_sign_ratio, on='phone')

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
        content_features = df.groupby('phone').agg({
            'message_length': ['mean', 'std', 'min', 'max'],
            'word_count': ['mean', 'std'],
            'digit_count': ['mean', 'sum'],
            'punctuation_count': ['mean', 'sum'], 
            'amount_count': ['sum'],
            'loan_keyword_count': ['sum', 'mean'],
            'risk_keyword_count': ['sum', 'mean'],
            'promo_keyword_count': ['sum', 'mean']
        }).reset_index()
        content_features.columns = ['phone'] + ['_'.join(col) for col in content_features.columns[1:]]
        return content_features

    def create_phone_area_features(self, df):
        df['area_code'] = df['phone'].str[3:7] 
        df['carrier_code'] = df['phone'].astype(str).str[:3]
        carrier_stats = df.groupby('carrier_code').agg({
            'phone': 'nunique',
            'message': 'count'
        }).reset_index()
        carrier_stats.columns = ['carrier_code', 'carrier_user_count', 'carrier_message_count']
        df = df.merge(carrier_stats, on='carrier_code', how='left')
        return df

    def create_messages_distribution_features(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df['contain_slash'] = df['message'].str.count('/')
        df['contain_url'] = df['message'].str.count('://')
        user_features = df.groupby('phone').agg({
            'message': ['count'],
            'sign': ['nunique'],
            'datetime': ['min', 'max'],
            'contain_slash': ['sum'],
            'contain_url': ['sum']
        }).reset_index()
        user_features.columns = ['phone', 'message_count', 'sign_unique_count', 'datetime_min', 'datetime_max', 'slash_count', 'url_count']
        user_features['time_span_days'] = (user_features['datetime_max'] - user_features['datetime_min']).dt.total_seconds() / (24 * 3600)
        user_features['time_span_days'] = user_features['time_span_days'].replace(0, 1/24)
        user_features['message_frequency'] = user_features['message_count'] / user_features['time_span_days']
        today = datetime.datetime.now()
        user_features['days_since_last_message'] = (today - user_features['datetime_max']).dt.total_seconds() / (24 * 3600)
        user_features['days_since_last_message'] = user_features['days_since_last_message'].replace(0, 1/24)
        user_features = user_features.drop(['datetime_min', 'datetime_max', 'time_span_days'], axis=1)
        return user_features

    def create_comprehensive_features(self, df):
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        time_features = self.create_time_features(df)
        content_features = self.create_message_content_features(df)
        sign_features = self.create_sign_diversity_features(df)
        df_with_area = self.create_phone_area_features(df)
        area_features = df_with_area.groupby('phone').agg({
            'carrier_user_count': 'first',
            'carrier_message_count': 'first'
        }).reset_index()
        messages_distribution_features = self.create_messages_distribution_features(df)
        all_features = time_features
        feature_dfs = [content_features,sign_features,area_features,messages_distribution_features]
        for i, feature_df in enumerate(feature_dfs):
            all_features = all_features.merge(feature_df, on='phone', how='left')
        self.logger.info(f"人工统计特征维度: {all_features.shape}")
        for col in all_features.columns:
            if all_features[col].isnull().sum()>0:
                all_features[col] = all_features[col].fillna(0)
        return all_features

    # def concat_extra_features(self, df, extra_features_df):
    #     df['phone'] = df['phone'].astype(str)
    #     extra_features_df['phone'] = extra_features_df['phone'].astype(str)
    #     merged_df = df.merge(
    #         extra_features_df, 
    #         left_on='phone', 
    #         right_on='phone', 
    #         how='inner'
    #     )
    #     return merged_df

    def get_top_n_signs(self, x, n=3):
        if x.dropna().empty:
            return 'unknown'
        sign_counts = x.value_counts()
        top_signs = sign_counts.head(n).index.tolist()
        return '|'.join(top_signs)

    def preprocess_batch(self, 
                         data: pd.DataFrame, 
                         data_tag: str, 
                         is_training: bool=True, 
                         chunk_id: int|None=None):
        
        today = datetime.datetime.now().strftime('%Y%m%d')
        df_label = None
        if is_training:
            df_label = data[['phone', 'label']]

        if self.filter_messages_setting:
            data = self.filter_messages(data=data, filter_messages_setting=self.filter_messages_setting)

        # generate static vector
        df_static_vec = data.copy()
        df_static_vec['text'] = df_static_vec['message']
        df_merged_static_vec = df_static_vec.groupby('phone').agg({
            'text': lambda x: ' '.join(x),
        }).reset_index()

        if is_training:
            if self.static_vec == 'tf-idf':
                self.logger.info(f'Using TF-IDF vectorizer')
                vectorizer = TfidfVectorizer(
                    tokenizer=self.tokenizer,
                    max_features=300, min_df=2, max_df=0.95, 
                    ngram_range=(1, 2), sublinear_tf=True
                )
                static_matrix = vectorizer.fit_transform(df_merged_static_vec['text'])
                try:
                    static_matrix = static_matrix.toarray()
                except Exception:
                    static_matrix = np.array(static_matrix)

                joblib.dump(vectorizer, f'{self.root_path}/model/static_vec/{data_tag}/{self.static_vec}_model.pkl')
                self.logger.info(f'Static vectorizer saved: {self.root_path}/model/static_vec/{data_tag}/{self.static_vec}_model.pkl')

            elif self.static_vec == 'word2vec':
                self.logger.info(f'Using Word2Vec vectorizer')
                all_messages = df_static_vec['message'].tolist()
                sentences = [jieba.lcut(msg) for msg in all_messages if msg.strip()]
                vectorizer = Word2Vec(
                    sentences=sentences, 
                    vector_size=300, 
                    window=5, 
                    min_count=2, 
                    workers=4,
                    epochs=10,  
                    sg=1  
                )
                tfidf_vectorizer = TfidfVectorizer(
                    tokenizer=self.tokenizer,
                    min_df=2, max_df=0.95
                )
                tfidf_vectorizer.fit(df_merged_static_vec['text'])
                self.logger.info(f'Using TF-IDF for Word2Vec weighted vectors')
                user_vectors = []
                for phone in df_merged_static_vec['phone']:
                    user_text = df_merged_static_vec[df_merged_static_vec['phone'] == phone]['text'].iloc[0]
                    user_words = jieba.lcut(user_text)
                    tfidf_scores = tfidf_vectorizer.transform([user_text]).toarray()[0]
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    word_to_tfidf = dict(zip(feature_names, tfidf_scores))
                    weighted_vectors = []
                    weights = []
                    for word in user_words:
                        if word in vectorizer.wv and word in word_to_tfidf:
                            weighted_vectors.append(vectorizer.wv[word])
                            weights.append(word_to_tfidf[word])
                    if weighted_vectors:
                        user_vector = np.average(weighted_vectors, weights=weights, axis=0)
                    else:
                        user_vector = np.zeros(300)
                    user_vectors.append(user_vector)
                static_matrix = np.array(user_vectors)
                vectorizer.save(f'{self.root_path}/model/static_vec/{self.static_vec}_model.bin')
                joblib.dump(tfidf_vectorizer, f'{self.root_path}/model/static_vec/{self.static_vec}_weighted_tfidf_model.pkl')
                self.logger.info(f'Word2Vec and TF-IDF models saved')
        else:
            if self.static_vec == 'tf-idf':
                vectorizer = joblib.load(f'{self.root_path}/model/static_vec/{self.static_vec}_model.pkl')
                self.logger.info(f'Static vectorizer loaded: {self.root_path}/model/static_vec/{self.static_vec}_model.pkl')
                static_matrix = vectorizer.transform(df_merged_static_vec['text'])
                if hasattr(static_matrix, 'toarray'):
                    static_matrix = static_matrix.toarray()
            elif self.static_vec == 'word2vec':
                vectorizer = Word2Vec.load(f'{self.root_path}/model/static_vec/{self.static_vec}_model.bin')
                tfidf_vectorizer = joblib.load(f'{self.root_path}/model/static_vec/{self.static_vec}_weighted_tfidf_model.pkl')
                self.logger.info(f'Word2Vec and TF-IDF models loaded')
                user_vectors = []
                for phone in df_merged_static_vec['phone']:
                    user_text = df_merged_static_vec[df_merged_static_vec['phone'] == phone]['text'].iloc[0]
                    # 分词方式根据配置选择
                    user_words = self.tokenizer(user_text)
                    tfidf_scores = tfidf_vectorizer.transform([user_text]).toarray()[0]
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    word_to_tfidf = dict(zip(feature_names, tfidf_scores))
                    weighted_vectors = []
                    weights = []
                    for word in user_words:
                        if word in vectorizer.wv and word in word_to_tfidf:
                            weighted_vectors.append(vectorizer.wv[word])
                            weights.append(word_to_tfidf[word])
                    if weighted_vectors:
                        user_vector = np.average(weighted_vectors, weights=weights, axis=0)
                    else:
                        user_vector = np.zeros(300)
                    user_vectors.append(user_vector)
                static_matrix = np.array(user_vectors)

        # 确保 static_matrix 为 numpy 数组
        if not isinstance(static_matrix, np.ndarray):
            try:
                static_matrix = static_matrix.toarray()
            except Exception:
                static_matrix = np.array(static_matrix)
        df_static_vec = pd.DataFrame(static_matrix, columns=[f'static_vec_{i}' for i in range(static_matrix.shape[1])])
        df_static_vec['phone'] = df_merged_static_vec['phone']
        del df_merged_static_vec, static_matrix
        gc.collect()

        # 动态特征处理
        df_dynamic_vec = data.copy()
        if self.dynamic_vec is not None:
            if self.dynamic_vec == 'bert':
                tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                embedding_model = BertModel.from_pretrained('bert-base-chinese')
            elif self.dynamic_vec == 'roberta':
                tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
                embedding_model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
            else:
                tokenizer = None
                embedding_model = None
            if tokenizer is not None and embedding_model is not None:
                embedding_model.eval()
                embedding_model.to(self.device)
                messages = df_dynamic_vec['message'].tolist()
                dynamic_embeddings = self.encode_texts_batch(
                    messages,
                    tokenizer=tokenizer,
                    model=embedding_model,
                    is_training=is_training
                )
                dynamic_embeddings_array = np.array(dynamic_embeddings)
                dynamic_embedding_dim = dynamic_embeddings_array.shape[1]
                self.logger.info(f'Embedding dim: {dynamic_embedding_dim}')
                df_embedding = pd.DataFrame(dynamic_embeddings_array, columns=[f'embedding_{i}' for i in range(dynamic_embedding_dim)])
                df_dynamic_vec = pd.concat([df_dynamic_vec.reset_index(drop=True), df_embedding], axis=1)
                embedding_columns = [col for col in df_dynamic_vec.columns if col.startswith('embedding_')]
                df_for_pooling = df_dynamic_vec[['phone'] + embedding_columns].copy()
                df_dynamic_vec = self.embedding_pooling(df_for_pooling)
                del messages, dynamic_embeddings, dynamic_embeddings_array, df_embedding, df_for_pooling
                del embedding_model, tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        else:
            # 如果未指定动态特征，直接用 phone 列
            df_dynamic_vec = df_dynamic_vec[['phone']].copy()

        # 其它特征
        df_features_extra = self.create_comprehensive_features(df)
        df_features = self.concat_embedding_tfidf(df_dynamic_vec, df_static_vec)
        df_features = self.concat_extra_features(df_features, df_features_extra)
        del df_static_vec, df_dynamic_vec, df_features_extra, df
        gc.collect()
        if is_training and df_label is not None:
            df_features = df_features.merge(
                df_label,
                on='phone',
                how='left'
            )
            self.logger.info(f'Training data merged with label, final shape: {df_features.shape}')
        if chunk_id is not None:
            output_path = f'{self.root_path}/data/{data_tag}_preproceed_{today}_chunk_{chunk_id}.csv'
        else:
            output_path = f'{self.root_path}/data/{data_tag}_preproceed_{today}.csv'
        df_features.to_csv(output_path, index=False)
        self.logger.info(f'Feature data saved: {output_path}')
        return output_path


    def preprocess(self, 
                   data_path: str, 
                   data_tag: str, 
                   enc_col: str|None = None,
                   chunk_size: int = 200000):  
        
        is_training = False
        file_name = data_path.split('/')[-1].split('.')[0]
        self.logger.info(f'Starting preprocessing for {file_name}')

        data = pd.read_csv(data_path, encoding='utf-8', on_bad_lines='skip')

        if self.use_aes and enc_col:
            self.logger.info(f'Using AES encryption for column: {enc_col}, transforming to column: "id" ')
            data = self.decrypt_data(data=data, enc_col=enc_col)

        if 'label' in data.columns:
            is_training = True
            df_label = data[['phone', 'label']]
            self.logger.info(f"Detected 'label' column, setting is_training=True")

        total_rows = len(data)
        if total_rows > chunk_size:
            self.logger.info(f'Total rows: {total_rows}, preprocessing in chunks of {chunk_size} rows')
            chunk_files = []
            num_chunks = (total_rows + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_rows)
                self.logger.info(f"Processing chunk {i+1}/{num_chunks}, row range: {start_idx}-{end_idx}")
                data_chunk = data.iloc[start_idx:end_idx].copy()

                chunk_file = self.preprocess_batch(
                    data=data_chunk,
                    data_tag=data_tag,
                    is_training=is_training,
                    chunk_id=i
                )
                
                chunk_files.append(chunk_file)
                
                # 清理内存
                del df_chunk
                gc.collect()
                
                self.logger.info(f"第 {i+1} 块处理完成")
            
            # 合并所有块的结果
            self.logger.info("开始合并所有块的结果...")
            combined_df = None
            
            for i, chunk_file in enumerate(chunk_files):
                chunk_df = pd.read_csv(chunk_file)
                
                if combined_df is None:
                    combined_df = chunk_df
                else:
                    combined_df = pd.concat([combined_df, chunk_df], ignore_index=True)
                
                # 删除临时文件
                os.remove(chunk_file)
                self.logger.info(f"已合并第 {i+1} 块，删除临时文件: {chunk_file}")
                
                del chunk_df
                gc.collect()



        if is_training:
            # training data contain labels

            df_label = data[['phone', 'label']]
            df_label.drop_duplicates(subset=['phone'], inplace=True)
            
            self.preprocess_batch(
                data=data,
                data_tag=data_tag,
                is_training=is_training,
            )
        else:
            # 预测模式：分块处理
            df = pd.read_csv(data_path, encoding='utf-8', names=['phone_id', 'message','sign','datetime'],on_bad_lines='skip')
            df = self.decrypt_df(df)
            
            total_rows = len(df)
            self.logger.info(f"总数据量: {total_rows}, 将分成 {chunk_size} 行一个块进行处理")
            
            chunk_files = []
            num_chunks = (total_rows + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_rows)
                self.logger.info(f"处理第 {i+1}/{num_chunks} 块数据，行数范围: {start_idx}-{end_idx}")

                df_chunk = df.iloc[start_idx:end_idx].copy()
                
                # 处理该块数据
                chunk_file = self.preprocess_batch(
                    data=df_chunk,
                    data_tag=data_tag,
                    is_training=is_training,
                    chunk_id=i
                )
                
                chunk_files.append(chunk_file)
                
                # 清理内存
                del df_chunk
                gc.collect()
                
                self.logger.info(f"第 {i+1} 块处理完成")
            
            # 合并所有块的结果
            self.logger.info("开始合并所有块的结果...")
            combined_df = None
            
            for i, chunk_file in enumerate(chunk_files):
                chunk_df = pd.read_csv(chunk_file)
                
                if combined_df is None:
                    combined_df = chunk_df
                else:
                    combined_df = pd.concat([combined_df, chunk_df], ignore_index=True)
                
                # 删除临时文件
                os.remove(chunk_file)
                self.logger.info(f"已合并第 {i+1} 块，删除临时文件: {chunk_file}")
                
                del chunk_df
                gc.collect()
            
            # 保存最终合并的结果
            today = datetime.datetime.now().strftime('%Y%m%d')
            final_output_path = f'{self.root_path}/data/{data_tag}_preproceed_{today}.csv'
            combined_df.to_csv(final_output_path, index=False)
            self.logger.info(f'最终合并数据保存完毕: {final_output_path}')
            self.logger.info(f'最终数据维度: {combined_df.shape}')
            
            del combined_df
            gc.collect()

