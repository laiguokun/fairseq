#!/bin/bash

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

GEN=$1
ori_l=$2
tgt_l=$3


SYS=$GEN.sys
REF=$GEN.ref
SYS_V2=$GEN.sys.v2
REF_V2=$GEN.ref.v2
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl

if [ $(tail -n 1 $GEN | grep BLEU | wc -l) -ne 1 ]; then
    echo "not done generating"
    exit
fi

grep ^H $GEN | cut -f3- | perl $TOKENIZER -threads 8 -a -l $ori_l > $SYS
grep ^T $GEN | cut -f2- | perl $TOKENIZER -threads 8 -a -l $tgt_l > $REF
echo "original bleu"
fairseq-score --sys $SYS --ref $REF
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $SYS > $SYS_V2
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $REF > $REF_V2
echo "modified bleu"
fairseq-score --sys $SYS_V2 --ref $REF_V2