import time
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from modules.ossl_classification import TrainModel ,AdMSoftmaxLoss, Model

if __name__ == '__main__':
    args = {
        'random_seed': 42,  # Random Seed
        'pretrained_model': "beomi/kcbert-large",  # Transformers PLM name
        'pretrained_tokenizer': "beomi/kcbert-large",
        # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
        'batch_size': 100,
        'lr': 5e-6,  # Starting Learning Rate
        'epochs': 20,  # Max Epochs
        'max_length': 150,  # Max Length input size
        'train_data_path': "../input/jytrain_ossl_50_revised.csv",  # Train Dataset file
        'val_data_path': "../input/jytest_ossl_50_revised.csv",  # Validation Dataset file
        'test_mode': False,  # Test Mode enables `fast_dev_run`
        'optimizer': 'AdamW',  # AdamW vs AdamP
        'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
        'fp16': True,  # Enable train on FP16
        'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
        'cpu_workers': os.cpu_count(),
    }
    start = time.time()
    train_model = TrainModel()
    train_model.main(args)
    print("time :", time.time() - start)