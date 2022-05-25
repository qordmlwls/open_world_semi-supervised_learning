import os	
import pandas as pd	
import torch	
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR	
from pytorch_lightning import LightningModule, Trainer, seed_everything	
from transformers import BertTokenizer, AdamW, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score	
import re	
import emoji	
from soynlp.normalizer import repeat_normalize
from torch.nn import functional as F
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
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
class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=10.0, m=0.4):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels,out):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        # self.fc = self.fc.cuda(x.device)
        # for W in self.fc.parameters():
        #     W = F.normalize(W, dim=1)

        # x = F.normalize(x, dim=1)

        # wf = self.fc(x)
        wf = out
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
## * 하나: 튜플 ** 둘: 딕셔너리	
class Model(LightningModule):	
    def __init__(self, **kwargs):	
        super().__init__()	
        self.save_hyperparameters()  # 이 부분에서 self.hparams에 위 kwargs가 저장된다.	
        self.num_classes = 100 ##긍부정 함 종류가 10정도라 정
        self.bert = BertModel.from_pretrained(self.hparams.pretrained_model)	
        self.layer = nn.Linear(1024,self.num_classes) # 500 - the number of pseudo-labels  
        self.labeled_num = 9 ## the number of labeled class 
        self.tokenizer = BertTokenizer.from_pretrained(	
            self.hparams.pretrained_tokenizer	
            if self.hparams.pretrained_tokenizer	
            else self.hparams.pretrained_model	
        )	
        self.u_bar = 0        

    def forward(self, **kwargs):	
        z = self.bert(**kwargs).pooler_output
        z = F.normalize(z,dim=-1)
        for W in self.layer.parameters():
            W = F.normalize(W, dim=-1)
        out = self.layer(z)
        weight = self.layer.weight
        return z,out , weight 

    ### unlabeled data - initialized with all 0
    def step(self, batch, batch_idx):	
        data, labels = batch
        unlabeled_data =[]
        unlabeled_label = []
        labeled_data = []
        labeled_label = []
        pseudo_label = []
        num_class = self.num_classes
        for dt,label in zip(data,labels):
            if label >= self.labeled_num   : ## 9 is num_labels
                unlabeled_data.append(dt)
                unlabeled_label.append(label)
            else:
                labeled_data.append(dt)
                labeled_label.append(label)
        z, out,weight = self(input_ids=data)	### to_do : labeled data도 만들어서 수정하기 
        prob = F.softmax(out,dim=-1)
        preds = prob.argmax(dim=-1)	## pseudo label 
        print(set(list(preds.cpu().numpy())))
        ## gernerate pseudo-label & compute cosine distance 제일 유사한 건 자기자신이므로 2번째로 유사한 것을 뽑는다 
        unlabeled_index = (labels == self.labeled_num).nonzero(as_tuple=False)
        
        try:
            idx = torch.topk(torch.Tensor(cosine_similarity(z.detach().cpu().numpy(),z.detach().cpu().numpy())),2).indices
            idx = [int(j) for i, j in idx]
            ## rearrange for finding most closet vector using torch.stack and list comprehension
        
            z_dasi_dasi = torch.stack([z[i] for i in idx],dim=0)
            out_dasi_dasi = torch.stack([out[i] for i in idx],dim=0)
            rearranged_labels = torch.stack([preds[j] if i in unlabeled_index else labels[j] for i, j  in enumerate(idx)],dim=0) ## rearrange psudo-label for pseudo-labeling
        except RuntimeError: ## only one element in batch 원래 case 나눠야하지만 한개 원소 케이스 거의없다 가정
            z_dasi_dasi = z.clone()
            rearranged_labels = preds.clone()
            out_dasi_dasi = out.clone()
        try:
            
            for i in range(len(unlabeled_data)):
                labels[unlabeled_index[i][0]] = rearranged_labels[unlabeled_index[i][0]] ## generate pseudo-labels 
        except IndexError: ## no unlabeled data
            pass
        if [unlabeled_data[i] for i in range(len(unlabeled_data))]:

            unlabeled_data = torch.stack([unlabeled_data[i] for i in range(len(unlabeled_data))],dim=0)
            unlabeled_label = torch.stack([unlabeled_label[i] for i in range(len(unlabeled_label))],dim=0)
            z_dasi, out_dasi , weight_dasi = self(input_ids=unlabeled_data)
            u = 1 - F.softmax(out_dasi,dim=-1).max(dim=-1).values
        else:
            unlabeled_data = torch.tensor([]).to(labels.device)
            unlabeled_label = torch.tensor([]).to(labels.device)
            out_dasi = torch.tensor([[0 for i in range(num_class)]]).to(labels.device) ## batch에 unlabeled data없는 경우
            u = torch.tensor([]).to(labels.device)

        amsoft2 =AdMSoftmaxLoss(in_features=1024,out_features=self.num_classes,m=-self.u_bar)
        if [labeled_data[i] for i in range(len(labeled_data))]:

            labeled_data = torch.stack([labeled_data[i] for i in range(len(labeled_data))],dim=0)
            labeled_label = torch.stack([labeled_label[i] for i in range(len(labeled_label))],dim=0)
            z_dasi_labeled, out_dasi_labeled , weight_dasi_labeled = self(input_ids=labeled_data)
            w_t_normal_labeled = F.normalize(weight_dasi_labeled)
            loss_s = amsoft2(z_dasi_labeled,labeled_label,out_dasi_labeled)

            ##labeled data만 정확도를 측정하도록 한다
            y_true = list(labeled_label.cpu().numpy())  
            y_pred = list(F.softmax(out_dasi_labeled,dim=-1).argmax(dim=-1).cpu().numpy())      
        else: ## there is no labeled data in the batch
            labeled_data = torch.tensor([]).to(labels.device)
            labeled_label = torch.tensor([]).to(labels.device)
            out_dasi_labeled = torch.tensor([[0 for i in range(num_class)]]).to(labels.device) ## batch에 labeled data없는 경우
            
            loss_s = 0
            
            y_true = [0]
            y_pred = [0]
        w_t = F.normalize(weight, dim=-1)
        z_normal = F.normalize(z, dim=-1)

        a = prob
        b = F.softmax(out_dasi_dasi,dim=-1)
        loss_p = torch.mean(-torch.log(torch.stack([torch.dot(a[i],b[i]) for i in range(len(a))])))
        
        prob_avg = torch.mean(prob,dim=0)
        ##@Todo : class_number parameter화 하기
        p = torch.ones(num_class).to(prob.device) / num_class
        regularization = -torch.sum(torch.log(prob_avg/p) * p)
        # Transformers 4.0.0+	
        loss = loss_s + 0.5*loss_p + 1.4*regularization	
        # logits = output.logits	
        #try log: 0.4,1.4로 학습함
        #try log: 1.0, 1.0으로 학습하니 하나의 클라스(7)만 나옴 -> regularization 필요
        self.log('train_loss',loss)
        
        return {	
            'loss': loss,	
            'y_true': y_true,	
            'u' : u,
            'y_pred': y_pred,	
        }	
    def training_step(self, batch, batch_idx):	
        return self.step(batch, batch_idx)	
    def validation_step(self, batch, batch_idx):	
        return self.step(batch, batch_idx)	
    def epoch_end(self, outputs, state='train'):	
        loss = torch.tensor(0, dtype=torch.float)	
        for i in outputs:	
            loss += i['loss'].cpu().detach()	
        loss = loss / len(outputs)	
        y_true = []	
        y_pred = []	
        u = []
        for i in outputs:	
            y_true += i['y_true']	
            try:
                u += i['u'].detach().cpu().numpy().tolist()
            except TypeError:
                u.append(i['u'].detach().cpu().numpy().tolist())
            # u.extend([i['u'].detach().cpu().numpy().tolist()])
            y_pred += i['y_pred']	
        print(set(y_pred))
        
        self.log(state + '_loss', float(loss), on_epoch=True, prog_bar=True)	
        self.log(state + '_acc', accuracy_score(torch.tensor(y_true,dtype=torch.float).cpu().numpy(), torch.tensor(y_pred,dtype=torch.float).cpu().numpy()), on_epoch=True, prog_bar=True)	
        self.log(state + '_precision', precision_score(torch.tensor(y_true,dtype=torch.float).cpu().numpy(), torch.tensor(y_pred,dtype=torch.float).cpu().numpy(), average='micro'), on_epoch=True, prog_bar=True)	
        self.log(state + '_recall', recall_score(torch.tensor(y_true,dtype=torch.float).cpu().numpy(), torch.tensor(y_pred,dtype=torch.float).cpu().numpy(), average='micro'), on_epoch=True, prog_bar=True)	
        self.log(state + '_f1', f1_score(torch.tensor(y_true,dtype=torch.float).cpu().numpy(), torch.tensor(y_pred,dtype=torch.float).cpu().numpy(), average='micro'), on_epoch=True, prog_bar=True)	
        # self.u_bar = np.mean(np.array(u)) ### 현재는 fit에 train epoch가 안돈다 -> evaluation epoch때 u_bar업데이트 함
        self.log(state + 'u_bar',np.mean(np.array(u)),on_epoch=True,prog_bar=True)
        return {'loss': loss,'u':u}	
    def training_epoch_end(self, outputs):	
        out = self.epoch_end(outputs,state='train')
        self.u_bar = np.mean(np.array(out['u']))
        # return out ##training_epoch_end는 return이 없어야된다 (validation_epoch는 리턴되는데..)
    def validation_epoch_end(self, outputs):	
        return self.epoch_end(outputs, state='val')	
    def configure_optimizers(self):	
        if self.hparams.optimizer == 'AdamW':	
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)	
        elif self.hparams.optimizer == 'AdamP':	
            from adamp import AdamP	
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)	
        else:	
            raise NotImplementedError('Only AdamW and AdamP is Supported!')	
        if self.hparams.lr_scheduler == 'cos':	
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)	
        elif self.hparams.lr_scheduler == 'exp':	
            scheduler = ExponentialLR(optimizer, gamma=0.5)	
        else:	
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')	
        return {	
            'optimizer': optimizer,	
            'scheduler': scheduler,	
        }	

    def read_data(self, path):	
        if path.endswith('xlsx'):	
            return pd.read_excel(path)	
        elif path.endswith('csv'):	
            return pd.read_csv(path,  lineterminator='\n')	
        elif path.endswith('tsv') or path.endswith('txt'):	
            return pd.read_csv(path, delimiter='\t', header=0, encoding="latin-1")	
        else:	
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def preprocess_dataframe(self, df):		
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())	
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')	
        url_pattern = re.compile(	
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')	
        url_pattern2 = re.compile(r'http[s]?\s?(com|kr)?\w*')
        url_pattern3 = re.compile(r'w{3}?\s?(com|kr)?\w*')
        phone_pattern= re.compile(r'010\s?[\w|\d]{4}\s?[\w|\d]{4}')
        email_pattern = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
        num = re.compile(r'([일|이|삼|사|오|육|칠|팔|구|십]*\s?[십|백|천|만|억]+\s?)+')
        def rm_emoji(Data):
            return Data.encode('utf-8','ignore').decode('utf-8')	
        def clean(x):
            x = str(x)
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
            x = rm_emoji(x)
            return x
        df=  df.drop_duplicates(['document'])
        df = df[df['document'].notnull()]
        ## label 전부 10으로 통일(unlabeled)
        df['label'] = df['label'].fillna(self.labeled_num)

        df['document'] = df['document'].map(lambda x: self.tokenizer.encode(	
            clean(str(x)),	
            padding='max_length',	
            max_length=self.hparams.max_length,	
            truncation=True,	
        ))	
        return df	
    def dataloader(self, path, shuffle=False):	
        df = self.read_data(path)	
        df = self.preprocess_dataframe(df)	
        dataset = TensorDataset(	
            torch.tensor(df['document'].to_list(), dtype=torch.long),	
            torch.tensor(df['label'].to_list(), dtype=torch.long),	
        )	
        return DataLoader(	
            dataset,	
            batch_size=self.hparams.batch_size or self.batch_size,	
            shuffle=shuffle,	
            num_workers=self.hparams.cpu_workers,	
        )	
    def train_dataloader(self):	
        return self.dataloader(self.hparams.train_data_path, shuffle=True)	
    def val_dataloader(self):	
        return self.dataloader(self.hparams.val_data_path, shuffle=False)	

class TrainModel:
    def __init__(self):
        self.checkpoint_callback = ModelCheckpoint(
            filename='epoch{epoch}-val_acc{val_acc:.4f}',
            monitor='val_acc',
            save_top_k=3,
            mode='max',
            # auto_insert_metric_name=False,
        )
    def main(self, args):
        print("Using PyTorch Ver", torch.__version__)
        print("Fix Seed:", args['random_seed'])
        seed_everything(args['random_seed'])
        model = Model(**args)
        print(":: Start Training ::")
        trainer = Trainer(
            callbacks=[self.checkpoint_callback],
            max_epochs=args['epochs'],
            fast_dev_run=args['test_mode'],
            num_sanity_val_steps=None if args['test_mode'] else 0,

            # For GPU Setup
            deterministic=torch.cuda.is_available(),
            gpus=-1 if torch.cuda.is_available() else None,
            precision=16 if args['fp16'] else 32,
            accelerator='dp',
            #plugins ='deepspeed'
            # For TPU Setup
            # tpu_cores=args.tpu_cores if args.tpu_cores else None,
        )
        trainer.fit(model)
# checkpoint_callback = ModelCheckpoint(
#     filename='epoch{epoch}-val_acc{val_acc:.4f}',
#     monitor='val_acc',
#     save_top_k=3,
#     mode='max',
#     # auto_insert_metric_name=False,
# )
# def main():
#     print("Using PyTorch Ver", torch.__version__)
#     print("Fix Seed:", args['random_seed'])
#     seed_everything(args['random_seed'])
#     model = Model(**args)
#     # checkpoint_path ='./lightning_logs/version_30707/checkpoints/epochepoch=1-val_accval_acc=0.4173.ckpt'
#     # checkpoint_path = "./lightning_logs/version_30746/checkpoints/epochepoch=3-val_accval_acc=0.6437.ckpt"
#     # model = model.load_from_checkpoint(checkpoint_path=checkpoint_path)
#     print(":: Start Training ::")
#     trainer = Trainer(
#         callbacks=[checkpoint_callback],
#         max_epochs=args['epochs'],
#         fast_dev_run=args['test_mode'],
#         num_sanity_val_steps=None if args['test_mode'] else 0,
#
#         # For GPU Setup
#         deterministic=torch.cuda.is_available(),
#         gpus=-1 if torch.cuda.is_available() else None,
#         precision=16 if args['fp16'] else 32,
#         accelerator='dp',
#         # plugins = DeepSpeedPlugin(stage=3,cpu_offload=True)
#         #plugins ='deepspeed'
#         # For TPU Setup
#         # tpu_cores=args.tpu_cores if args.tpu_cores else None,
#     )
#     trainer.fit(model)
# import time
# start = time.time()
# if __name__ == '__main__':
#     main()
#     print("time :", time.time() - start)