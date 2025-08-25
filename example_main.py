# example_main.py

from src.data_processor import DataProcessor
from src.trainer import Trainer
from src.predictor import Predictor
import pandas as pd
import joblib



if __name__ == "__main__":
    # source data path
    data_path = "data/raw/train_data.csv"

    # AES encryption settings
    use_aes = True
    aes_key = b'20250703qwerzxcv'
    aes_iv = b'20250703qwerzxcv'

    # Message filter settings:
    # keywords_to_remove: Messages containing any of these keywords will be removed.
    # keep_keywords: Messages containing any of these keywords have the highest priority and will never be deleted under any circumstances.
    filter_messages_setting = {
        "keywords_to_remove": ['快递', '外卖', '美食', '包裹', '物流'],
        "keep_keywords": ['贷', '借', '金融', '信用', '好分期', '花呗', '呗', '钱']
    }

    processor = DataProcessor(
        use_aes=use_aes,
        aes_key=aes_key,
        aes_iv=aes_iv,
        set_cn_stopwords=True,
        filter_messages_setting=filter_messages_setting,
        static_vec="word2vec",
        dynamic_vec='roberta',
        language='cn'
    )

    processed_path = processor.preprocess(data_path=data_path, 
                                        data_tag="demo", 
                                        chunk_size=200000)

    # # 训练
    # processed_df = pd.read_csv(processed_path)
    # from sklearn.ensemble import RandomForestClassifier
    # X = processed_df.drop(['phone', 'label'], axis=1)
    # y = processed_df['label']
    # trainer = Trainer(model=RandomForestClassifier(), X=X, y=y)
    # model = trainer.train()
    # trainer.save("model/demo_rf.joblib")

    # # 推理
    # predictor = Predictor("model/demo_rf.joblib")
    # preds = predictor.predict(X)
    # print("预测结果示例：", preds[:10])