import glob
import os


class DataBuilder(object):

    def __init__(self, corpus):
        self.corpus = corpus

    @staticmethod
    def read_note(file_dir):
        with open(file_dir, 'r') as f:
            text = f.read()

        return text

    def read_dir(self, sub_dir):
        examples = []
        full_dir = os.path.join(self.corpus, sub_dir)
        for file_dir in glob.glob(full_dir):
            text = self.read_note(file_dir)
            examples.append(text)
        return examples

    def get_data(self):
        examples_train_yes = self.read_dir(sub_dir='train/yes/*.txt')
        examples_train_no = self.read_dir(sub_dir='train/no/*.txt')
        examples_test_yes = self.read_dir(sub_dir='test/yes/*.txt')
        examples_test_no = self.read_dir(sub_dir='test/no/*.txt')

        examples_train = examples_train_yes + examples_train_no
        labels_train = len(examples_train_yes) * [1] + len(examples_train_no) * [0]

        examples_test = examples_test_yes + examples_test_no
        labels_test = len(examples_test_yes) * [1] + len(examples_test_no) * [0]

        return examples_train, labels_train, examples_test, labels_test
