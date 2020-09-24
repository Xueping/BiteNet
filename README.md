# BiteNet

## Data Preparation
dataset.data_prepararion.py for MIMIC III dataset

## Train Model
BiteNet.train.BiteNet_mh_DX.py for future diagnosis prediction.

BiteNet.train.BiteNet_mh_RE.py for future re-admission prediction.

## Model Parameters
--data_source mimic3 

--model Bite 

--verbose True 

--task BiteNet 

--predict_type dx 

--visit_threshold 2  

--max_epoch 5 

--train_batch_size 32 

--gpu 2 

--valid_visits 10 

--num_hidden_layers 1 

--pos_encoding encoding 

--min_cut_freq 5 

--embedding_size 128 

--dropout 0.1 

--only_dx_flag False