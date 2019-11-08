TEXT=data-bin/wmt16_en_de_spm
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.spm --validpref $TEXT/newstest2013.spm --testpref $TEXT/newstest2014.spm \
    --destdir data-bin/wmt16_en_de_spm_bin \
    --joined-dictionary --srcdict $TEXT/vocab.spm.fairseq \
    --workers 20