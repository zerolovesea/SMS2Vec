import json
from src.trainer import Trainer
from src.config_manager import ConfigManager

if __name__ == "__main__":

    # load sign_id_vocab
    with open("data/resources/sign_id_map.json", "r") as f:
        sign_id_vocab = json.load(f)

    sign_vocab_size = max(sign_id_vocab.values()) + 1

    config = {
        'data_path': "data/preproceed/train_qwen3_w2v_mean_sign_seq_hfq_credit.csv",
        'model_tag': 'DNN_QWEN3_W2V_mean_sign_seq_HFQ_ABSEED',
        'label_col': 'label',
        'batch_size': 64,
        'epochs': 50,
        'patience': 10, # early stopping patience
        'lr': 1e-3,
        'model_params': {
            'hidden_dims': [128, 64, 32],
            'activation': 'relu', # dice|prelu|softmax|linear
            'dropout': 0.1,
            'dice_dim': 2,
            'use_sign_embedding': True,
            'sign_seq_len': 50,
            'sign_vocab_size': sign_vocab_size,
            'sign_embedding_dim': 32,
        },

        'device': 'cpu'
    }


    trainer = Trainer(config)
    trainer.train()
