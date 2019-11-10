#!/bin/bash

# source shflags
. ~/shflags/shflags

# define a 'name' command-line string flag
DEFINE_string 'bias' 'False' 'whether has bias' 'b'
DEFINE_string 'rel_attn' 'False' 'use relative attention' 'r'
DEFINE_string 'model' '' 'model_name without .pt, assume it is located in tf folder' 'm'
DEFINE_string 'gpu' '0' 'use which gpu' 'g'
DEFINE_string 'l' '6' 'layer_num' 'l'
DEFINE_string 'dim' '1024' 'hidden_dim' 'd'
DEFINE_string 'a' '16' 'head_num' 'a'
# parse the command-line
FLAGS "$@" || exit $?
eval set -- "${FLAGS_ARGV}"

model=${FLAGS_model}
l=${FLAGS_l}
h=${FLAGS_dim}
a=${FLAGS_a}
ffn_h=$[$h * 4]

bool_args=""
if [ "${FLAGS_bias}" = "True" ]; then
  bool_args="${bool_args} --softmax-bias"
fi
if [ "${FLAGS_rel_attn}" = "True" ]; then
  bool_args="${bool_args} --relative-attn"
fi


