# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import fairseq
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
random.seed(1984)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ' + str(device))

########## Please change these to your own paths ##########

datadir = '/home/smg/cooper/phase1-main/DATA'
cp_path = '/home/smg/cooper/proj-mosnet-phase2/fairseq-pretrained-models/wav2vec2/w2v_large_lv_fsh_swbd_cv.pt'
## ^ path to a pretrained fairseq model.
## please use a wav2vec_small.pt, w2v_large_lv_fsh_swbd_cv.pt, or xlsr_53_56k.pt

###########################################################

wavdir = os.path.join(datadir, 'wav')
trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
validlist = os.path.join(datadir, 'sets/val_mos_list.txt')

if cp_path.split('/')[-1] == 'wav2vec_small.pt':
    SSL_OUT_DIM = 768
else:
    SSL_OUT_DIM = 1024

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
ssl_model = model[0]
ssl_model.remove_pretraining_modules()

class MosPredictor(nn.Module):
    def __init__(self, ssl_model):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = SSL_OUT_DIM
        self.output_layer = nn.Linear(self.ssl_features, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x = self.output_layer(x)
        return x.squeeze(1)

    
class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = { }
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            self.mos_lookup[wavname] = mos

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys())

        
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        score = self.mos_lookup[wavname]
        return wav, score, wavname
    

    def __len__(self):
        return len(self.wavnames)


    def collate_fn(self, batch):  ## zero padding
        wavs, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames
    

trainset = MyDataset(wavdir, trainlist)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

validset = MyDataset(wavdir, validlist)
validloader = DataLoader(validset, batch_size=2, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

net = MosPredictor(ssl_model)
net = net.to(device)

criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

if __name__ == '__main__':
    PREV_VAL_LOSS=9999999999
    orig_patience=20
    patience=orig_patience
    for epoch in range(1,1001):
        STEPS=0
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))

        ## have it run validation every epoch & print loss
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        ## validation
        VALSTEPS=0
        for i, data in enumerate(validloader, 0):
            VALSTEPS+=1
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()

        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            PATH = './checkpoints/ckpt_' + str(epoch)
            torch.save(net.state_dict(), PATH)
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Training')

