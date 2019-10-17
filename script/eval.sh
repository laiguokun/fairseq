#!/bin/sh

# source shflags
. ~/shflags/shflags

# define a 'name' command-line string flag
DEFINE_string 'bias' 'False' 'whether has bias' 'b'
DEFINE_string 'model' '' 'model_name without .pt, assume it is located in tf folder' 'm'
DEFINE_string 'gpu' '0' 'use which gpu' 'g'

# parse the command-line
FLAGS "$@" || exit $?
eval set -- "${FLAGS_ARGV}"

model=${FLAGS_model}
if [ "${FLAGS_bias}" = "False" ]; then
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
      --snap_model_file ./tf/$model.pt \
      --only_convert \
      --save-dir ./save/$model
else
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
      --snap_model_file ./tf/$model.pt \
      --softmax_bias \
      --only_convert \
      --save-dir ./save/$model
fi


echo "validation"

gpu=${FLAGS_gpu}
CUDA_VISIBLE_DEVICES=$gpu python validate.py \
    data-bin/wmt16_en_de_bpe32k \
    --task translation --dataset-impl mmap \
    --path checkpoints/$model.pt --max-tokens 4000 \
    --user-dir models \
    
echo "test set evaluation"
CUDA_VISIBLE_DEVICES=$gpu python generate.py \
    data-bin/wmt16_en_de_bpe32k \
    --task translation --dataset-impl mmap\
    --path checkpoints/$model.pt --max-tokens 4000\
    --user-dir models \
    --beam 5 --remove-bpe --lenpen 0.35 --gen-subset test > wmt16_gen.txt
    
bash ./scripts/compound_split_bleu.sh wmt16_gen.txt