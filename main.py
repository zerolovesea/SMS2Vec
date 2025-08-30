# example_main.py

import pandas as pd
from src.config_manager import ConfigManager
from src.data_processor import DataProcessor




if __name__ == "__main__":
    processing_config = ConfigManager("processing_config.yaml")

    # source data path
    train_data_path = "data/raw/messages.csv" 
    # pred_data_path = "data/raw/predict_data.csv"

    use_aes = processing_config.get('use_aes', False)
    aes_key = processing_config.get('aes_key', None)

    if aes_key is not None and isinstance(aes_key, str):
        aes_key = aes_key.encode('utf-8')
    aes_iv = processing_config.get('aes_iv', None)
    if aes_iv is not None and isinstance(aes_iv, str):
        aes_iv = aes_iv.encode('utf-8')

    filter_messages_setting = processing_config.get('filter_messages_setting', {})
    static_vec = processing_config.get('static_vec', None) # word2vec|tf-idf
    dynamic_vec = processing_config.get('dynamic_vec', None) # qwen3|bge-m3|roberta
    tf_idf_config = processing_config.get('tf_idf_config', {})
    w2v_config = processing_config.get('w2v_config', {})
    dynamic_vec_pooling_method = processing_config.get('dynamic_vec_pooling_method', 'mean') # mean|max|min|sum
    messages_keywords_config = processing_config.get('messages_keywords_config', {})
    sign_id_sequences_max_len = processing_config.get('sign_id_sequences_max_len', 50)

    processor = DataProcessor(
        use_aes=use_aes,
        aes_key=aes_key,
        aes_iv=aes_iv,
        filter_messages_setting=filter_messages_setting,
        static_vec="word2vec",
        w2v_config=w2v_config,
        tf_idf_config=tf_idf_config,
        dynamic_vec='qwen3',
        dynamic_vec_pooling_method=dynamic_vec_pooling_method,
        messages_keywords_config=messages_keywords_config,
        sign_id_sequences_max_len=sign_id_sequences_max_len,
        language='cn',
    )

    train_data_processed_path = processor.preprocess(data_path=train_data_path, 
                                        enc_col='phone_id', 
                                        data_tag='demo_train_qwen3_w2v_mean_sign_sequences', 
                                        chunk_size=200000)
    
    # pred_data_processed_path = processor.preprocess(
    #                                     data_path=pred_data_path, 
    #                                     enc_col='enc_id', 
    #                                     data_tag='demo_qwen3_w2v_pred', 
    #                                     chunk_size=200000)

    # processed_df = pd.read_csv(processed_path)

