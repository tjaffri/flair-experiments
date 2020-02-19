# Adapted from https://github.com/zalandoresearch/flair/blob/master/resources/docs/EXPERIMENTS.md
import argparse
from typing import List
import os


from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from tqdm import tqdm

MINI_BATCH_SIZE = 32
HIDDEN_SIZE = 256

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='The directory of data')
    parser.add_argument('--train_file', type=str, default='train.iobes',
                        help='The name of the train file, under --data_dir')
    parser.add_argument('--dev_file', type=str, default='dev.iobes',
                        help='The name of the dev file; under --data_dir')
    parser.add_argument('--test_file', type=str, default='test.iobes',
                        help='The name of the test file; under --data_dir')
    parser.add_argument('--forward_flair_embeddings', type=str, default='news-forward',
                        help='Forward flair embeddings')
    parser.add_argument('--backward_flair_embeddings', type=str, default='news-backward',
                        help='Backward flair embeddings')
    parser.add_argument('--test_output_file', type=str, default='test_output.iobes',
                        help='The name of the file with test output; under --data_dir')
    parser.add_argument('-lr', '--learning_rate_find', action='store_true', default=False,
                        help='Don\'t train. Only plot learning rate.')

    return parser.parse_args()


def tag_and_output(sents, tagger, path, tag_type):
    with open(path, 'w') as f:
        for sent in tqdm(sents):
            gold_tags = [token.get_tag(tag_type).value for token in sent]
            tagger.predict(sent)
            predicted_tags = [token.get_tag(tag_type).value for token in sent]
            for token, gold, predicted in zip(sent, gold_tags, predicted_tags):
                f.write(f'{token.text} {predicted}\n')

            f.write('\n') # blank line between sentences


def main():
    args = parse_args()

    if not os.path.exists(args.data_dir):
        raise Exception(f'Path does not exist: {args.data_dir}')

    # 1. Build corpus
    columns = {0: 'text', 1: 'ner'}
    corpus: Corpus = ColumnCorpus(args.data_dir, columns,
                                  train_file=args.train_file,
                                  dev_file=args.dev_file,
                                  test_file=args.test_file)

    print(corpus)
    print(corpus.obtain_statistics())

    # 2. What tag do we want to predict?
    tag_type = 'ner'

    # 3. Build tag dictionary
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    # 4. Initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('crawl'),
        FlairEmbeddings(args.forward_flair_embeddings),
        FlairEmbeddings(args.backward_flair_embeddings),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. Initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=HIDDEN_SIZE,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type)

    # 6. Initialize trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    if args.learning_rate_find:
        print('***** Plotting learning rate')
        # 7a. Find learning rate
        learning_rate_tsv = trainer.find_learning_rate('temp', 'learning_rate.tsv', mini_batch_size = MINI_BATCH_SIZE)

    else:
        print('***** Running train')
        # 7b. Run Training
        trainer.train('temp',
                    learning_rate = 0.1,
                    mini_batch_size = MINI_BATCH_SIZE,
                    # it's a big dataset so maybe set embeddings_in_memory to False
                    embeddings_storage_mode='none')

        tag_and_output(corpus.test, tagger, os.path.join(args.data_dir, args.test_output_file), tag_type)


if __name__ == '__main__':
    main()
