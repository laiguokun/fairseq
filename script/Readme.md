## Eval Example

### convert the tf mdoel to pytorch model

Use the en_de_pretrain as the example
```
cd tf
bash convert.sh en_de_pretrain/model.ckpt-0 en_de_pretrain.pt
```

### eval the model

```
cd ..
bash eval.sh -m en_de_pretrain -b True
```
