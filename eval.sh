CUDA_VISIBLE_DEVICES=3 python validate_mul.py \
    data-bin/wmt14_en_fr \
    --task translation --dataset-impl mmap\
    --path checkpoints/wmt14.en-fr.joined-dict.transformer/model.pt --max-tokens 4000\
