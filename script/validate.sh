CUDA_VISIBLE_DEVICES=0 python validate.py \
    data-bin/wmt16_en_de_bpe32k \
    --task translation --dataset-impl mmap\
    --path checkpoints/en_de_pretrain.pt --max-tokens 4000\
    --user-dir models
