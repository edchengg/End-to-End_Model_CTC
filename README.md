# End-to-End Automatic Speech Recognition(ASR) CTC

This repository contains baseline models(3-5 layers Bi-LSTM) for ASR tasks on standard speech datasets(TIMIT, WSJ, Switchboard).

## Model
3 layers or 5 layers BiLSTM + Softmax Layer + CTC Loss

## Dataloader
3 dataloaders for 3 different datasets

## Results
|| Switchboard     | WSJ         | TIMIT  |
|:----- |:------- |:-------|:-----|
|Dev|11.86(CER)|6.1(CER)|13.429(PER)|
|Test||4.6(CER)|15.967(PER)|

## Visualization
Visualization of LSTM hidden units before pretraining and after pretraining.

![Alt Text](https://github.com/edchengg/End-to-End_Model_CTC/blob/master/img/5layers_pretrained_full_false.gif)
![Alt Text](https://github.com/edchengg/End-to-End_Model_CTC/blob/master/img/5layers_pretrained_full_true.gif)
