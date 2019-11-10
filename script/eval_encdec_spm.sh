#!/bin/sh

cur_dir="$(dirname "$0")"
. $cur_dir/shared_flags.sh

CUDA_VISIBLE_DEVICES= python train.py \
    data-bin/wmt16_en_de_spm_bin \
    --user-dir models \
    --arch my_transformer_wmt_en_de_big --language-embedding \
    --log-interval 100 --no-progress-bar \
    --max-update 30000 --share-all-embeddings --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --min-lr 1e-09 --update-freq 32 --keep-last-epochs 10 \
    --max-tokens 1800 \
    --lr-scheduler cosine --warmup-init-lr 1e-7 --warmup-updates 10000 \
    --lr-shrink 1 --max-lr 0.0009 --lr 1e-7 --min-lr 1e-9 --warmup-init-lr 1e-07 \
    --t-mult 1 --lr-period-updates 20000 \
    --snap_model_file ./tf/$model.pt \
    --encoder-attention-heads $a --decoder-attention-heads $a \
    --encoder-embed-dim $h --decoder-embed-dim $h \
    --encoder-ffn-embed-dim $ffn_h --decoder-ffn-embed-dim $ffn_h \
    --encoder-layers $l --decoder-layers $l \
    --only_convert \
    --save-dir ./save/$model \
    ${bool_args}

echo "===== NLL evaluation ====="
CUDA_VISIBLE_DEVICES=$FLAGS_gpu python validate.py \
    data-bin/wmt16_en_de_spm_bin \
    --task translation --dataset-impl mmap \
    --path checkpoints/$model.pt --max-tokens 4000 \
    --user-dir models

echo "===== BLEU evaluation ====="
CUDA_VISIBLE_DEVICES=$FLAGS_gpu python generate.py \
    data-bin/wmt16_en_de_spm_bin \
    --task translation --dataset-impl mmap \
    --path checkpoints/$model.pt --max-tokens 4000 \
    --user-dir models \
    --sacrebleu \
    --beam 1 --remove-bpe sentencepiece --lenpen 0.6 --gen-subset valid > wmt16_gen.txt

bash script/spm_bleu.sh wmt16_gen.txt en de
