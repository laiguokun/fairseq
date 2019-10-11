TEXT=examples/translation/wmt14_en_fr
fairseq-preprocess \
    --source-lang en --target-lang fr --joined-dictionary --srcdict ./checkpoints/wmt14.en-fr.joined-dict.transformer/dict.en.txt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_fr \
    --workers 32