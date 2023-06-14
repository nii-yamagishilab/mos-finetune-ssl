# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

## run inference without requiring ground-truth answers
## or system info.

import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from mos_fairseq import MosPredictor, MyDataset
import numpy as np
import scipy.stats
import datetime
import time

def unixnow():
    return str(int(time.mktime(datetime.datetime.now().timetuple())))


def systemID(uttID):
    return uttID.split('-')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model.')
    parser.add_argument('--datadir', type=str, required=True, help='Path of your directory containing .wav files')
    parser.add_argument('--finetuned_checkpoint', type=str, required=True, help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--outfile', type=str, required=False, default='answer.txt', help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
    args = parser.parse_args()
    
    cp_path = args.fairseq_base_model
    my_checkpoint = args.finetuned_checkpoint
    wavdir = args.datadir
    outfile = args.outfile

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))

    wavfnames = [x for x in os.listdir(wavdir) if x.split('.')[-1] == 'wav']
    wavlist = 'tmp_' + unixnow() + '.txt'
    wavlistf = open(wavlist, 'w')
    for w in wavfnames:
        wavlistf.write(w + ',3.0\n')
    wavlistf.close()

    print('Loading data')
    validset = MyDataset(wavdir, wavlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    total_loss = 0.0
    num_steps = 0.0
    predictions = { }  # filename : prediction
    criterion = nn.L1Loss()
    print('Starting prediction')

    for i, data in enumerate(validloader, 0):
        inputs, labels, filenames = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        output = outputs.cpu().detach().numpy()[0]
        predictions[filenames[0]] = output  ## batch size = 1


    ## generate answer.txt for codalab
    ans = open(outfile, 'w')
    for k, v in predictions.items():
        outl = k.split('.')[0] + ',' + str(v) + '\n'
        ans.write(outl)
    ans.close()

    os.system('rm ' + wavlistf)

if __name__ == '__main__':
    main()
