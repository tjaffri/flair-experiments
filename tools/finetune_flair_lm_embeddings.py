# Adapted from https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md

import argparse

from os.path import isdir
from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

DEFAULT_BASE_MODEL = 'news-forward'

def fine_tune(base_model, corpus_dir, output_dir):

    # print stats
    print(f'Fine tuning base model: {base_model}')
    print(f'Corpus dir: {corpus_dir}')
    print(f'Output dir: {output_dir}')

    # instantiate an existing LM, such as one from the FlairEmbeddings
    language_model = FlairEmbeddings(base_model).lm

    # are you fine-tuning a forward or backward LM?
    is_forward_lm = language_model.is_forward_lm

    # get the dictionary from the existing language model
    dictionary: Dictionary = language_model.dictionary

    # get your corpus, process forward and at the character level
    corpus = TextCorpus(corpus_dir,
                        dictionary,
                        is_forward_lm,
                        character_level=True)

    # use the model trainer to fine-tune this model on your corpus
    trainer = LanguageModelTrainer(language_model, corpus)

    trainer.train(output_dir,
                sequence_length=100,
                mini_batch_size=100,
                learning_rate=20,
                patience=10,
                checkpoint=True)



def parse_args():
    """
    Parses command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Fine-tunes flair language model embeddings on a given corpus dir (must contain valid flair corpus i.e. text.txt / valid.txt plus a train dir with train splits)')

    parser.add_argument('-f', '--corpus_dir', help='Path to corpus dir where corpus files can be found')
    parser.add_argument('-o', '--output_dir', help='Path to output dir where model should be saved')
    parser.add_argument('-b', '--base_model', help='Model to fine tune. Default: ' + DEFAULT_BASE_MODEL, required=False)

    args = parser.parse_args()
    return args

def main():
    """
    Main entrypoint of the CLI
    """
    args = parse_args()

    # validate arguments
    if not args.corpus_dir or not isdir(args.corpus_dir):
        raise Exception(f'Invalid corpus_dir: {args.corpus_dir}')
    
    if not args.output_dir or not isdir(args.output_dir):
        raise Exception(f'Invalid output_dir: {args.output_dir}')
    
    base_model = DEFAULT_BASE_MODEL
    if args.base_model:
        base_model = args.base_model

    fine_tune(base_model, args.corpus_dir, args.output_dir)

if __name__ == '__main__':
    main()
