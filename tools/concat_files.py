#!/usr/bin/env python3

"""
Copyright (c) 2019 Docugami Inc.
"""

import argparse
import re
import os
from os.path import isdir, join

from tqdm import tqdm

DEFAULT_MAX_CHARS_PER_SINGLE_FILE = 1e7
FILE_EXTENSION = '.txt'

def read_text_file(file_path, encoding='utf-8'):
    """
    Reads a text file from a given path and returns a string with its contents.
    """
    file_content = ''
    with open(file_path, 'r', encoding=encoding) as file_handle:
        file_content = file_handle.read()

    return file_content

def concat_files(source_dir, target_dir, base_target_filename, create_splits, max_chars_per_split):

    # scan for files
    print(f'Scanning for files...')
    file_paths = []
    for dir_path, _, files in os.walk(source_dir):
        for fname in files:
            if fname.lower().endswith(FILE_EXTENSION):
                input_file_path = join(dir_path, fname)
                file_paths.append(input_file_path)

    # initialize
    target_filename = None
    current_split_id = 1
    current_split_len = 0
    if create_splits:
        target_dir = join(target_dir, base_target_filename)
        os.mkdir(target_dir)

    # process
    print(f'Processing files...')
    for file_path in tqdm(file_paths):
        input_file_contents = read_text_file(file_path)
        current_split_len += len(input_file_contents)

        # open target file in append mode
        target_filename = f'{base_target_filename}{FILE_EXTENSION}'
        if create_splits:
            target_filename = f'{base_target_filename}_{current_split_id}{FILE_EXTENSION}'
        with open(join(target_dir, target_filename), 'a+', encoding='utf-8') as file_handle:
            file_handle.write(input_file_contents + '\n')

        # bump the split ID if we have reached the limit
        if create_splits and current_split_len > DEFAULT_MAX_CHARS_PER_SINGLE_FILE:
            current_split_len = 0
            current_split_id += 1


def parse_args():
    """
    Parses command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Traverses the given source dir for .txt files, and concats them to the given file name in the given target dir (optionally, creates splits).')

    parser.add_argument('-f', '--from_dir', help='Path to source dir where .txt files can be found.')
    parser.add_argument('-t', '--target_dir', help='Path to target dir where the flair format corpus will be created.')
    parser.add_argument('-b', '--base_target_filename', help='Name of target file name (without extension), e.g. "test" or "valid" or "train".')
    parser.add_argument('-s', '--create_splits', help='Create splits in the target dir (under a new dir called the value of base_target_filename).', action='store_true', default=False)
    parser.add_argument('-m', '--max_chars_per_split', help='Max chars per split file', required=False)

    args = parser.parse_args()
    return args

def main():
    """
    Main entrypoint of the CLI
    """
    args = parse_args()

    # validate arguments
    if not args.from_dir or not isdir(args.from_dir):
        raise Exception(f'Invalid from_dir: {args.from_dir}')
    
    if not args.target_dir or not isdir(args.target_dir):
        raise Exception(f'Invalid target_dir: {args.target_dir}')
    
    if not args.base_target_filename:
        raise Exception(f'Invalid base_target_filename: {args.base_target_filename}')
    
    max_chars_per_split = DEFAULT_MAX_CHARS_PER_SINGLE_FILE
    if args.max_chars_per_split:
        max_chars_per_split = args.max_chars_per_split

    concat_files(args.from_dir, args.target_dir, args.base_target_filename, args.create_splits, max_chars_per_split)

if __name__ == '__main__':
    main()
