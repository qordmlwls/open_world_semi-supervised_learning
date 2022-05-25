import os	
import pandas as pd	
import numpy as np
from pytorch_lightning import plugins
from pytorch_lightning.utilities import distributed
from sklearn.utils.multiclass import check_classification_targets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertModel
import torch	
from torch.utils.data import Dataset, DataLoader, TensorDataset	
from torch.optim.lr_scheduler import ExponentialLR	
from pytorch_lightning import LightningModule, Trainer
import re	
import emoji	
from soynlp.normalizer import repeat_normalize	
import time	
import unicodedata
from multiprocessing import Pool
import warnings
from collections import OrderedDict
# from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from torch.nn import functional as F
import torch.nn as nn
warnings.filterwarnings("ignore")

class Model(LightningModule):
    def __init__(self, options):
        super().__init__()
        self.args = options
        self.bert = BertModel.from_pretrained(self.args.pretrained_model, num_labels=100)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.pretrained_tokenizer if self.args.pretrained_tokenizer else self.args.pretrained_model
        )
        self.num_classes = 100 ##긍부정 함 종류가 10정도라 정
        self.layer = nn.Linear(1024,self.num_classes) # 500 - the number of pseudo-labels  
        # self.bert = BertForSequenceClassification.from_pretrained(self.hparams.pretrained_model)	
        self.Dropout = nn.Dropout(p=0.1,inplace=False)
        self.labeled_num = 9
        # self.tokenizer = AutoTokenizer.from_pretrained(	
        #     self.hparams.pretrained_tokenizer	
        #     if self.hparams.pretrained_tokenizer	
        #     else self.hparams.pretrained_model	
        # )	
    def forward(self, **kwargs):	
        z = self.bert(**kwargs).pooler_output
        z = F.normalize(z,dim=-1)
        for W in self.layer.parameters():
            W = F.normalize(W, dim=-1)
        out = self.layer(z)
        weight = self.layer.weight
        ## weight = w.transpose()
        return z,out , weight 

    # 이때의 테스트 스텝은 각 배치별로 도는 것임
    def test_step(self, batch, batch_idx):  # 멀티 GPU 사용 -> 각 배치를 3개의 GPU 에 할당
        """
        :param batch: 배치 크기 데이터 뭉치
        :return: 로짓 기반의 예측 y 텐서 리스트 (각 배치의 텐서 -> 이것들이 리스트로 모임)
        """
        # data, labels = batch
        data, labels = batch
        output = self(input_ids=data)  # output: 예측된 결과의 logits 을 포함
        logits = output.logits
        _predicts = logits.argmax(dim=-1)  # 각 행의 로짓을 argmax 로 예측값을 기록
        return {'y_pred': _predicts}  # 각 배치의 예측값 tensor 이 저장
    # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
    #     return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
    def predict_step(self, batch, batch_idx):
        data, labels = batch
        num_class = self.num_classes
        z, out,weight = self(input_ids=data)	### to_do : labeled data도 만들어서 수정하기 
        _prob = F.softmax(out,dim=-1)
        _predicts = _prob.argmax(dim=-1)	## pseudo label 
        # logits = output.logits
        # _predicts = logits.argmax(dim=-1)
        prob = F.softmax(out).max(dim=-1)[0]
        # output = OrderedDict({
        #     'y_pred': _predicts.detach().cpu().numpy().tolist(),
        #     # "batch_size": len(labels)
        #     })
        # output = _predicts.detach().cpu().numpy().tolist()
        output = OrderedDict({
            'prob': prob.detach().cpu().numpy().tolist(),
            'y_pred': _predicts.detach().cpu().numpy().tolist(),
            "labels": labels.detach().cpu().numpy().tolist()
            })
        return output

        
    def test_epoch_end(self, outputs):
        """
        :param outputs: dictionary
        outputs[i] : epoch size (i: df size / batch size ; ex: 60000/ 256 -> 235 )
        outputs[i]['y_pred'] : predictions for each epoch

        :return: y_pred : list  ; list of predictions
        """
        
        y_pred = []
        for i in range(len(outputs)):  # range of the epochs (whole df size / batch size = epoch size)
            y_pred.extend(list(outputs[i]['y_pred'].cpu().numpy()))
            # y_pred.append(outputs[i]['y_pred'].cpu().numpy().tolist())
        return {'y_pred': y_pred}
    # def test_epoch_end(self, outputs):
    #     # embs = []
    #     # for i in range(len(outputs)):
    #     #     embs.append(outputs[i]['embedding'])
    #     # embs = torch.cat(embs)
    #     # out_emb = [torch.zeros_like(embs) for _ in range(dist.get_world_size())]
    #     # dist.barrier()
    #     # dist.all_gather(out_emb, embs)
    #     # if dist.get_rank() == 0:
    #     #     interleaved_out = torch.empty((embs.shape[0]*dist.get_world_size(), embs.shape[1]), device=embs.device, dtype=embs.dtype)
    #     #     for current_rank in range(dist.get_world_size()):
    #     #         interleaved_out[current_rank::dist.get_world_size()] = out_emb[current_rank]
    #     #     interleaved_out = interleaved_out[:len(dataset)]
        
    #     if torch.distributed.is_initialized():
    #         torch.distributed.barrier()
    #         gather = [None] * torch.distributed.get_world_size()
    #         torch.distributed.all_gather_object(gather, outputs)
    #         outputs = [x for xs in gather for x in xs]
    #     # global outputs2
    #     # outputs2 = outputs
    #     # print(outputs2)
    #     return outputs
            # Create KDTree and compute recall

    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'AdamP':
            optimizer = AdamP(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')

        if self.args.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        # lr_scheduler 는 exp 로 고정인데 코드상에서 분기가 있는 이유는?
        # elif self.args.lr_scheduler == 'cos':
        #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')

        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

class SnsContentClassifier:
    def __init__(self):
        self.args = None
        self.content_df = None
        # global i 
    def load_model(self):
        # 21 categories
        # checkpoint_path = '../bert_model/epochepoch=2-val_accval_acc=0.9997.ckpt'    
        # checkpoint_path = './lightning_logs/version_30677/checkpoints/epochepoch=1-val_accval_acc=0.7574.ckpt'
        # checkpoint_path = "./lightning_logs/version_30707/checkpoints/epochepoch=1-val_accval_acc=0.4173.ckpt"
        checkpoint_path = "./lightning_logs/version_30772/checkpoints/epochepoch=5-val_accval_acc=0.7141.ckpt"
        pretrained_model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path, options=self.args)
        pretrained_model = pretrained_model.to("cuda")
        pretrained_model.eval()
        # pretrained_model.freeze()
        return pretrained_model
    # multiprocess split with cpu_count of num cores
    @staticmethod
    def parallelize_dataframe(df, func):
        num_cores = os.cpu_count()
        # num_cores = 8
        if len(df) < num_cores:
            num_cores = len(df)
        df_split = np.array_split(df, num_cores)
        pool = Pool(num_cores)
        concatenated = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return concatenated
    @staticmethod
    def multiprocessing_func_preprocess(df):
        # preprocess functions
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        url_pattern2 = re.compile(r'http[s]?\s?(com|kr)?\w*')
        url_pattern3 = re.compile(r'w{3}?\s?(com|kr)?\w*')
        phone_pattern= re.compile(r'010\s?[\w|\d]{4}\s?[\w|\d]{4}')
        email_pattern = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
        num = re.compile(r'([일|이|삼|사|오|육|칠|팔|구|십]*\s?[십|백|천|만|억]+\s?)+')
        tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-large")
        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip('\n')
            x = email_pattern.sub('',x)
            x = phone_pattern.sub('번호',x)
            x = url_pattern2.sub(' ', x)
            x = url_pattern3.sub(' ', x)
            x = num.sub('숫자',x)
            x = unicodedata.normalize('NFC',x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x
        df.rename(columns={"content": "document"}, inplace=True)
        
        # map function for each row
        df['document'] = df['document'].map(
            lambda x: tokenizer.encode(clean(str(x)), padding='max_length', max_length= 150, truncation=True)
        )

        return df

    def preprocess_dataframe(self, df):
        pre_time = time.time()
        _df = self.parallelize_dataframe(df, self.multiprocessing_func_preprocess)  # multiprocess with cpu
        print(f'Elapsed for preprocess data: {round(time.time() - pre_time, 3)} seconds')
        return _df

    def test_dataloader(self):
        load_time = time.time()
        _df = self.preprocess_dataframe(self.content_df)

        # create label column w/o any value for testing
        # _df['label'] = 0
        dataset = TensorDataset(
            torch.tensor(_df['document'].to_list(), dtype=torch.long),
            torch.tensor([i for i in range(len(_df['document'].to_list()))], dtype=torch.long),
        )
        print(f'Elapsed for data loading(read+preprocess): {round(time.time() - load_time, 3)} seconds')
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers
        )

    # TRAINING THE TRAINER FUNCTION
    def training(self, pretrained_model):
        train_start = time.time()
        # Trainer
        trainer = Trainer(
            max_epochs=self.args.epochs,
            fast_dev_run=self.args.test_mode,
            num_sanity_val_steps=None if self.args.test_mode else 0,
            auto_scale_batch_size=self.args.auto_batch_size if self.args.auto_batch_size and not self.args.batch_size else False,
            # For GPU Setup
            deterministic=torch.cuda.is_available(),
            gpus=-1,
            accelerator='ddp' if torch.cuda.is_available() else None,
            # distributed_backend='ddp',
            # plugins=DDPPlugin(),
            # num_nodes=1,
        
            # gpus=[2,3],
            precision=16 if self.args.fp16 else 32,
        )

        # train_result = trainer.test(pretrained_model, test_dataloaders=self.test_dataloader(),verbose=False)
        train_result = trainer.predict(pretrained_model,dataloaders=self.test_dataloader())
        # train_result = trainer.predict(pretrained_model, dataloaders=self.test_dataloader())
        print(f'Elapsed for KcBert to test: {round(time.time() - train_start, 3)} seconds')
        # return train_result
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            gather = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gather, train_result)
            # outputs = sum([x for xs in gather for x in xs],[])
            outputs = [x for xs in gather for x in xs]
            prob = sum([k['prob'] for k in outputs],[])
            y_pred = sum([k['y_pred'] for k in outputs],[])
            orders = sum([z['labels'] for z in outputs],[])
            tmp_dic = OrderedDict({ name:value for name, value in zip(orders, y_pred) })
            tmp_dic2 = OrderedDict({ name:value for name, value in zip(orders, prob) })
            sorted_dic = sorted(tmp_dic.items()) ##순서를 다시 맞춰줌
            sorted_dic2 = sorted(tmp_dic2.items())
            result = [value for key,value in sorted_dic]
            result2 = [value for key,value in sorted_dic2]
        return result, result2    # result: 예측값

    def true_y(self, x):
        if x == 0:
            return '거래판매'
        elif x == 1:
            return '렌탈'
        elif x == 2:
            return '부동산'
        elif x == 3:
            return '수리'
        elif x == 4:
            return '이벤트'
        elif x == 5:
            return '인사글'
        elif x == 6:
            return '종교'
        elif x == 7:
            return '주식'
        elif x == 8:
            return '체험단'
        else:
            return '진성'
    
    def mapping_process(self, content_df, train_result, train_result2):
        start_time = time.time()
        # content_df['y_hat'] = train_result[0]['y_pred']
        content_df['prob'] = train_result2
        content_df['y_hat'] = train_result
        content_df['y_hat_label'] = content_df['y_hat'].map(lambda x: self.true_y(x))

        print(f'Time taken for mapping_process {round(time.time() - start_time, 3)} seconds')

        return content_df  # 모델 예측값 + 룰베이스 예측값: 최종결과값
    def run(self, content_df):
        class Arg:
            random_seed: int = 42
            # Transformers PLM name
            pretrained_model: str = 'beomi/kcbert-large'
            # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
            pretrained_tokenizer: str = ''
            # Let PyTorch Lightening find the best batch size
            auto_batch_size: str = 'power'
            # Optional, Train/Eval Batch Size. Overrides `auto_batch_size`
            batch_size: int = 256
            # Starting Learning Rate
            lr: float = 5e-6
            # Max Epochs
            epochs: int = 20
            # Max Length input size
            max_length: int = 150
            # Report (Train Metrics) Cycle
            report_cycle: int = 100
            # Multi cpu workers
            cpu_workers: int = os.cpu_count()
            # KcBERT_Garbage2.0 Mode enables `fast_dev_run`
            test_mode: bool = False
            optimizer: str = 'AdamW'
            lr_scheduler: str = 'exp'
            # Enable train on FP16
            fp16: bool = True
            # Enable TPU with 1 core or 8 cores
            tpu_cores: int = 0

        self.args = Arg()
        self.content_df = content_df

        pretrained_model = self.load_model()

        train_result, train_result2 = self.training(pretrained_model)

        result_df = self.mapping_process(content_df=content_df,train_result=train_result, train_result2=train_result2)

        return result_df
        # return train_result

