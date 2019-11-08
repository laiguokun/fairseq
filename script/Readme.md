## Eval Example

### convert the tf mdoel to pytorch model

Use the en_de_pretrain as the example
```
cd tf
bash convert.sh -i en_de_pretrain -o en_de_pretrain.pt (-l 14 -d 1024)
```

### eval the model

```
cd ..
bash eval.sh -m en_de_pretrain -b True (-l 14 -d 1024)
```

## Eval Encoder-Decoder model

### convert the tf mdoel to pytorch model

Use the ende_nobias_L12H768A12 as the example
```
cd tf
bash convert_encdec.sh -i ende_nobias_L12H768A12 -o ende_nobias_L12H768A12.pt (-l 12 -d 768)
```

### eval the model

```
cd ..
bash eval_encdec.sh -m ende_nobias_L12H768A12 -l 12 -d 768 -a 12
```

## Eval Encoder-Decoder model with SPM

### convert the tf mdoel to pytorch model

Use the ende_nobias_L12H768A12 as the example
```
cd tf
bash convert_encdec.sh -i ende_nobias_L12H768A12 -o ende_nobias_L12H768A12.pt -l 12 -d 768 -v 32000
```

### eval the model

```
cd ..
bash eval_encdec_spm.sh -m ende_nobias_L12H768A12 -l 12 -d 768 -a 12
```
