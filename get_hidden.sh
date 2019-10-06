CUDA_VISIBLE_DEVICES=2,3 python get_hidden.py \
    data-bin/wmt14_en_fr \
    --task translation --dataset-impl mmap\
    --path checkpoints/wmt14.en-fr.joined-dict.transformer/model.pt --max-tokens 4000\
