# ERA1 Session17

This repository contains the source code to the transformer model (that supports the variants such as BERT, GPT, ViT)

base module: [transformer.py][def]


[def]: transformer/transformer.py

Refer to the [bert.ipynb][def2], [gpt.ipynb][def3] and [vit_2.ipynb][def4] for the implementation of BERT, GPT and ViT model respectively.


[def2]: Assignment_solution/S17_assign_BERT.ipynb
[def3]: Assignment_solution/S17_assign_GPT.ipynb
[def4]: Assignment_solution/S17_Assign_ViT.ipynb

#### Training Logs for BERT
```
training...
it: 0  | loss 10.2  | Δw: 1.136
it: 10  | loss 9.55  | Δw: 0.558
it: 20  | loss 9.38  | Δw: 0.326
it: 30  | loss 9.16  | Δw: 0.248
it: 40  | loss 9.04  | Δw: 0.208
it: 50  | loss 8.84  | Δw: 0.184
it: 60  | loss 8.69  | Δw: 0.17
it: 70  | loss 8.56  | Δw: 0.158
it: 80  | loss 8.31  | Δw: 0.146
it: 90  | loss 8.22  | Δw: 0.144
it: 100  | loss 8.04  | Δw: 0.125
it: 110  | loss 7.96  | Δw: 0.116
it: 120  | loss 7.75  | Δw: 0.116
it: 130  | loss 7.65  | Δw: 0.109
it: 140  | loss 7.54  | Δw: 0.113
it: 150  | loss 7.43  | Δw: 0.104
it: 160  | loss 7.35  | Δw: 0.094
it: 170  | loss 7.25  | Δw: 0.091
it: 180  | loss 7.08  | Δw: 0.087
it: 190  | loss 7.01  | Δw: 0.093
it: 200  | loss 6.96  | Δw: 0.087
it: 210  | loss 6.9  | Δw: 0.089
it: 220  | loss 6.87  | Δw: 0.083
it: 230  | loss 6.71  | Δw: 0.082
...
it: 960  | loss 6.25  | Δw: 0.903
it: 970  | loss 6.17  | Δw: 0.892
it: 980  | loss 6.19  | Δw: 0.95
it: 990  | loss 6.16  | Δw: 0.943
```

#### Tranining Logs for GPT
```
step          0 | train loss 10.7437 | val loss 10.7569
step        499 | train loss 0.3488 | val loss 8.0341
```

#### Traning Logs for ViT(Using Pretrained model)
```
Epoch: 1 | train_loss: 0.7756 | train_acc: 0.7305 | test_loss: 0.5107 | test_acc: 0.8873
Epoch: 2 | train_loss: 0.3257 | train_acc: 0.9336 | test_loss: 0.3085 | test_acc: 0.8674
Epoch: 3 | train_loss: 0.2190 | train_acc: 0.9492 | test_loss: 0.2568 | test_acc: 0.8674
```

#### Traning Logs for ViT(Training from scratch)
```
Epoch: 1 | train_loss: 3.9789 | train_acc: 0.4141 | test_loss: 2.3860 | test_acc: 0.5417
Epoch: 2 | train_loss: 1.7153 | train_acc: 0.2656 | test_loss: 1.1288 | test_acc: 0.5417
Epoch: 3 | train_loss: 1.2529 | train_acc: 0.2930 | test_loss: 1.0564 | test_acc: 0.5417
Epoch: 4 | train_loss: 1.1921 | train_acc: 0.4023 | test_loss: 1.0075 | test_acc: 0.5417
```