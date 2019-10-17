SAVE=save/$1
CUDA_VISIBLE_DEVICES= python train.py \
    data-bin/wmt16_en_de_bpe32k \
    --user-dir models \
    --arch joint_attention_wmt_en_de_big \
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
    --snap_model_file ./tf/$1.pt \
    --only_convert \
    --save-dir $SAVE
