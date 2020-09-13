# Unified Transformer
![](https://i.imgur.com/67luDQW.png)

# Introduction 
In this repository, we implement Unified Transformer for IWSLT and WMT dataset. Unified transformer is a unified bidirectional model not only focuses on learning the joint probability in two sequential domain mapping but also the integrating of domain knowledge and domain mapping. Our model builds a more complete learning scheme that exploits the information from given data.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* If no fairseq package, please install it as following instructions
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
git checkout 803c0a6d11fcca6dcff8b8d0a4170338b15b26ff 
```
* git clone this repository and you will have the five files 
    * dualformer.py 
    * dualformer_layer.py
    * label_smoothed_cross_entropy.py
    * fairseq_encoder.py
    * __init__.py
* Put the three files in the corresponding path in fairseq package 

```bash
fairseq/
├── docs/
├── examples/
├── build/
├── fairseq_cli
├── fairseq.egg-info
.
.
.

├── fairseq/
    ├── criterions/
        ├── label_smoothed_cross_entropy.py 
    ├── models/
        ├── dualformer.py     
        ├── fairseq_encoder.py
    ├── modules/
        ├── dualformer_layer.py
        ├── __init__.py
```
# Getting Started
## IWSLT De-En
For WMT De-En or En-De translation task, you can still following instruction and change prepare-iwslt14.sh to prepare-wmt14en2de.sh. 
And set a different path for the preprocessed data like 

### Preparing the dataset
All the instructions are the same as the fairseq package
```bash
cd examples/translation
bash prepare-iwslt14.sh
cd ../..

TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

### Train new De-En translation model
The default argument for all the hyperparameters are provided as following.
```bash
fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch dualformer_wmt_en_de  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0014 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --min-lr 1e-09 --warmup-init-lr 1e-07\
    --dropout 0.1 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --save-interval-updates  1000 \
    --keep-interval-updates 20 \
    --max-tokens 4506 \
    --update-freq 6 \
    --log-format json \
    --log-interval 50 \
    --max-epoch 35 --fp16 
```
## Evaluate well-trained model
```bash
fairseq-generate data-bin/iwslt14.tokenized.de-en/ \
                --path checkpoints_en_de_dualformer_iwslt/checkpoint_last.pt \
                --lenpen 0.6 \
                --beam 4 \
                --remove-bpe 
```

For other translation tasks that are not implemented in fairseq, you can just change some part of the script that prepares the data. 


## Experimental result
Comparing with standard transformer and other dual learning methods


| Model       | MLM | LM  | Reg    | Int1 | BLEU(De-En) | 
| ----------- | --- | --- | ------ | ---- | ----------- | 
| Unified transformer  | v   |     |        | v    | 33.60       |
| Unified transformer  | v   |     |        | v    | 33.01       |
| Unified transformer  | v   |     |        |      | 32.5        |
| Unified transformer  |     | v   | 0.001  |      | 31.31       |
| Unified transformer  |     | v   | 0.002  |      | 31.43       |
| Unified transformer  |     | v   | 0.0001 |      | 31.5        |
| Unified transformer  |     | v   | 0.0005 |      | 31.54       |
| Unified transformer  |     |    |        |      | 31.35       |
| Transformer |     |     |        |       | 30.08       |


| Model | BLEU(De-En) |
| -------- | -------- |
| Transformer + DSL | 32.5 |
| Transformer + Model-level dual| 33.2|
| Unified transformer | 34.01 |

BT - backtranslation

DSL - dual supervised learning 

Model-level dual - model-level dual learning

# Package Reference
[fairseq: A Fast, Extensible Toolkit for Sequence Modeling](https://www.aclweb.org/anthology/N19-4009.pdf)

# Related Work References
[Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

[Understanding Back-Translation at Scale](https://www.aclweb.org/anthology/D18-1045.pdf)

[Model-Level Dual Learning](http://proceedings.mlr.press/v80/xia18a/xia18a.pdf)

[Dual Supervised Learning](http://proceedings.mlr.press/v70/xia17a/xia17a.pdf)



