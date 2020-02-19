# Flair Experiments
Some experiments with the flair library: https://github.com/flairNLP/flair

# Pre-reqs

``bash
pip3 install -r ./requirements.txt
``

# BASIC: Train an NER model (with default flair embeddings)

Official Documentation: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md#training-a-sequence-labeling-model

1. First, you need some training data in IOBES format. For example, one of the datasets here: https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data (please comply with their posted license)

2. Train an NER model using the ``tools/train_ner_model.py script``. For example:

``bash
python3 tools/train_ner_model.py --data_dir ../../github/MTL-Bioinformatics-2016/data/BioNLP11EPI-IOBES/ \
                --train_file train.tsv --dev_file devel.tsv --test_file test.tsv 
``

When training is done, you will find your model find under ./temp. The test_output.iobes file (created next to your test file at the path give above) contains the model's predictions on your test file. Flair will also print out various scores and stats you can review.

To use the model yourself, try something like the code snippet in ``./test_ner.py``
``

# ADVANCED: Train an NER model (with pretrained embeddings fine tuned on your own corpus)

Official documentation: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md

## Create pretraining corpus
First, you need to reorganize your corpus in a very specific format that works for flair:

``bash
    valid.txt
    test.txt
    train
        train_1.txt
        train_2.txt
        ...
        train_n.txt
``

Assuming you have a bunch of *.txt files under a directory (organized however you want, e.g. in subfolders) you can use the ``tools/concat_files.py``
script to create the corpus above by reading *.txt files recursively and creating the concatenated files with the names as expected by flair. For the train files
you can create splits using the --create_splits option.

## Pretrain flair embeddings (by fine tuning some existing embeddings)

Now that you have your pretraining corpus, you can take some default flair embedding and fine tune on your own corpus:

``bash
python3 tools/finetune_flair_lm_embeddings.py --corpus_dir ./temp/corpus --output_dir ./temp/output --base_model news-forward
``

Depending on the size of your corpus the above can take a very long time. Use screen, knockknock, etc to 

You likely need to do the above twice, once with a forward model (e.g. news-forward) and once with a backward model (e.g. news-backward). Save the
output of both runs as separate model files.

## Train NER model with your pretrained flair embeddings
Now, you can retrain your NER model with your new pretrained flair embeddings. 

The only thing you need to do here is to provide the --forward_flair_embeddings and --backward_flair_embeddings parameters (paths to the two forward and backward models you fine tuned above). The rest of the instructions are identical to training an NER model with default flair embeddings. You should get a higher F1 score if you pretrained for some time, as compared to using default embeddings.

# License
MIT. See LICENSE file.
