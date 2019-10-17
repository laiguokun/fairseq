model=nobias_randinit
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python generate.py \
    data-bin/wmt16_en_de_bpe32k \
    --task translation --dataset-impl mmap\
    --path checkpoints/$model.pt --max-tokens 4000\
    --user-dir models \
    --beam 5 --remove-bpe --lenpen 0.35 --gen-subset test > wmt16_gen.txt
    
bash ./scripts/compound_split_bleu.sh wmt16_gen.txt