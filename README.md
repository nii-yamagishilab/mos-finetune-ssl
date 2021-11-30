# Finetune SSL models for MOS prediction

This is code for our paper under review for ICASSP 2022:

"Generalization Ability of MOS Prediction Networks"  Erica Cooper, Wen-Chin Huang, Tomoki Toda, Junichi Yamagishi  https://arxiv.org/abs/2110.02635

Please cite this preprint if you use this code.

## Dependencies:

 * Fairseq toolkit:  https://github.com/pytorch/fairseq  Make sure you can `import fairseq` in Python.
 * torch, numpy, scipy, torchaudio
 * I have exported my conda environment for this project to `environment.yml`
 * You also need to download a pretrained wav2vec2 model checkpoint.  These can be obtained here:  https://github.com/pytorch/fairseq/tree/main/examples/wav2vec  Please choose `wav2vec_small.pt`, `w2v_large_lv_fsh_swbd_cv.pt`, or `xlsr_53_56k.pt`. 
 * You also need to have a MOS dataset.  Datasets for the MOS prediction challenge will be released once the challenge starts.  TODO update with a link.

## How to use
 * Modify the paths in `mos_fairseq.py` to point to your own data and SSL checkpoints.
 * Run `python mos_fairseq.py` to finetune an SSL model on the data.
 * Modify variables in `predict.py` to point to your favorite checkpoint.
 * Run `predict.py` to run inference using that checkpoint.
 * A file called `answer.txt` should have been generated; you can compress it using zip and submit it to CodaLab.

## Acknowledgments

This study is supported by JST CREST grants JP- MJCR18A6, JPMJCR20D3, and JPMJCR19A3, and by MEXT KAKENHI grants 21K11951 and 21K19808. Thanks to the organizers of the Blizzard Challenge and Voice Conversion Challenge, and to Zhenhua Ling, Zhihang Xie, and Zhizheng Wu for answering our questions about past challenges.  Thanks also to the Fairseq team for making their code and models available.

## License

BSD 3-Clause License

Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
