# SSL-MOS: Baseline system of the first VoiceMOS Challenge

Author: Erica Cooper (National Institute of Informatics) Email: ecooper@nii.ac.jp

The finetuned SSL system implemented in this repository serves as one of the baselines of the first VoiceMOS Challenge, a challenge to compare different systems and approaches on the task of predicting the MOS score of synthetic speech.  In this challenge, we use the BVCC dataset.

## Training Phase (Phase 1)

During the training phase, the training set and the developement set are released. In the following, we demonstrate how to use a pretrained model to make predictions on the development set to generate a results file that can be submitted to the CodaLab platform, and how to train your own model from scratch.

### Dependencies

Please make sure you have installed the dependencies specified in the main README file for this repository.

### Data preparation

After downloading the dataset preparation scripts, please follow the instructions to gather the complete training and development set. For the rest of this README, we assume that the data is put under `data/`, but feel free to put it somewhere else. The data directorty should have the following structure:
```
data
└── phase1-main
    ├── DATA
    │   ├── mydata_system.csv
    │   ├── sets
    │   │   ├── DEVSET
    │   │   ├── train_mos_list.txt
    │   │   ├── TRAINSET
    │   │   └── val_mos_list.txt
    │   └── wav
    └─── ...
```

### Inference from pretrained model

We provide a pre-finetuned model on Dropbox; to download it and run inference, run:

```python run_inference_for_challenge.py --datadir data/phase1-main/DATA```

You should see the following output:

```DEVICE: cuda
Loading checkpoint
Loading data
Starting prediction
[UTTERANCE] Test error= 0.326597
[UTTERANCE] Linear correlation coefficient= 0.870618
[UTTERANCE] Spearman rank correlation coefficient= 0.871113
[UTTERANCE] Kendall Tau rank correlation coefficient= 0.692518
[SYSTEM] Test error= 0.186204
[SYSTEM] Linear correlation coefficient= 0.943495
[SYSTEM] Spearman rank correlation coefficient= 0.949144
[SYSTEM] Kendall Tau rank correlation coefficient= 0.810290```

A file called `answer.txt` will also be generated.

### Submission to CodaLab

The submission format of the CodaLab competition platform is a zip file (can be any name) containing a text file called `answer.txt` (this naming is a **MUST**).  
To submit to the CodaLab competition platform, compress `answer.txt` in zip format (via `zip` command in Linux or GUI in MacOS) and name it whatever you want. Then this zip file is ready to be submitted!

### Finetuning your own model

First, make sure you already have the dataset and one pretrained fairseq base model (e.g., `fairseq/wav2vec_small.pt`).

To run your own finetuning using the BVCC dataset, run:

```python mos_fairseq.py --datadir data/phase1-main/DATA --fairseq_base_model fairseq/wav2vec_small.pt```

Once the training has finished, checkpoints can be found in the `checkpoints` directory.  The best one is the one with the highest number.  To run inference using this checkpoint, run:

```python predict.py --fairseq_base_model fairseq/wav2vec_small.pt --finetuned_checkpoint checkpoints/ckpt_XX --datadir data/phase1-main/DATA ```