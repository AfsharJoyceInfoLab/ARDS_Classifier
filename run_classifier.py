#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import glob
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


def read_data(corpus):
    examples = []

    for file_name in glob.glob(os.path.join(corpus, '*.txt')):
        with open(file_name, 'r') as f:
            data = f.read()
            examples.append(data)
    return examples


if __name__ == '__main__':

    warnings.filterwarnings(action='ignore')

    parser = argparse.ArgumentParser()

    # models

    parser.add_argument("--cui_model_file", default="./models/cui_model.sav", type=str,
                        help='saved model trained on CUI data')
    parser.add_argument("--cui_vectorizer_file", default="./models/cui_vectorizer.sav", type=str,
                        help='saved tf-idf vectorizer')
    parser.add_argument("--text_model_file", default="./models/text_model.sav", type=str,
                        help='saved model trained on text data')
    parser.add_argument("--text_vectorizer_file", default="./models/text_vectorizer.sav", type=str,
                        help='saved tf-idf vectorizer')

    # data

    parser.add_argument("--cui_data_dir", default=None, type=str,
                        help='directory contains cui data')
    parser.add_argument("--text_data_dir", default=None, type=str,
                        help='directory contains text data')

    # output dir

    parser.add_argument("--output_dir", default=None, required=True, type=str,
                        help='directory to store predictions')

    args = parser.parse_args()

    # read data
    print('***Loading data***')
    examples = None
    if args.cui_data_dir is not None:
        examples = read_data(args.cui_data_dir)
        print('Finished loading.')
        print('Number of examples = ', len(examples))

        print('\n\n***Loading CUI model and vectorizer***')
        with open(args.cui_vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(args.cui_model_file, 'rb') as f:
            model = pickle.load(f)
        print('Finished loading.')
    else:
        examples = read_data(args.text_data_dir)
        print('Number of examples = ', len(examples))

        print('\n\n***Loading Text model and vectorizer***')
        with open(args.text_vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(args.text_model_file, 'rb') as f:
            model = pickle.load(f)
        print('Finished loading.')

    print('\n\n***Start predicting***')
    x = vectorizer.transform(examples)
    predicted_labels = model.predict(x)
    predicted_probs = model.predict_proba(x)[:, 1]
    print('Finished predicting.')

    print('\n\n***Start saving***')

    with open(os.path.join(args.output_dir, 'predicted_labels.txt'), 'w') as f:
        for i in predicted_labels.tolist():
            f.write(str(i))
            f.write('\n')

    with open(os.path.join(args.output_dir, 'predicted_probabilities.txt'), 'w') as f:
        for i in predicted_probs.tolist():
            f.write(str(i))
            f.write('\n')
    print('Finished saving')