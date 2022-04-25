# Homework 2 ADL NTU 110 Spring

## Download model
```shell
bash download.sh
```

## Context Selection
```shell
python train_context.py -m [pretrained model]
```

## Question Answering
```shell
python train_qa.py -m [pretrained model]
```

## Prediction
```shell
bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv
```

## Plot learning curve of training loss
```shell
python plot_loss.py --ckpt_dir /path/to/model
```

## Plot learning curve of EM
```shell
python plot_EM.py --context_dir /path/to/context.json --eval_dir /path/to/valid.json --context_ckpt_dir /path/to/context_model --qa_ckpt_dir /path/to/qa_model
```