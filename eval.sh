CUDA_VISIBLE_DEVICES=0 python validate_knn.py \
    data-bin/wmt14_en_fr \
    --task translation --dataset-impl mmap \
    --topk 128 \
    --lamb 0.1 \
    --path checkpoints/wmt14.en-fr.joined-dict.transformer/model.pt --max-tokens 4000\
