#!/bin/sh

# source shflags
. ~/shflags/shflags

# define a 'name' command-line string flag
DEFINE_string 'out' '' 'output_file' 'o'
DEFINE_string 'inp' '' 'input_folder' 'i'
DEFINE_string 'l' '14' 'layer_num' 'l'
DEFINE_string 'dim' '1024' 'hidden_dim' 'd'
DEFINE_string 'vocab' '32768' 'vocab_size' 'v'

# parse the command-line
FLAGS "$@" || exit $?
eval set -- "${FLAGS_ARGV}"

inp=${FLAGS_inp}
out=${FLAGS_out}
l=${FLAGS_l}
h=${FLAGS_dim}
v=${FLAGS_vocab}

python ckpt_to_npy.py --input_ckpt $inp/model.ckpt-0
python npy_to_pt.py $out $l $h $v
