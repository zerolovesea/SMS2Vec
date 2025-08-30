import json
from src.trainer import Trainer


if __name__ == "__main__":
    with open("data/resources/sign_id_map.json", "r") as f:
        sign_id_vocab = json.load(f)
    sign_vocab_size = len(sign_id_vocab)

    config = {
        'data_path': "data/preproceed/demo_train_qwen3_w2v_mean_sign_sequences.csv",
        'model_tag': 'MLP_BGE_M3_W2V_mean_no_sign_sequences',
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
            'use_sign_embedding': False,
            'sign_seq_len': 50,
            'sign_vocab_size': sign_vocab_size,
            'sign_embedding_dim': 32,
        },

        'device': 'mps'
    }
    trainer = Trainer(config)
    trainer.train()
