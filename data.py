import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = self.process_line(line) #altered
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = self.process_line(line) #altered
                ids = [self.dictionary.word2idx[word] for word in words]
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

    # Added by me
    def process_line(self, line):
        """Processes a line of text, appending <eos> after specific punctuation."""
        punctuations = {'.', '?', '!'}
        words = line.split()
        processed_words = []
        for word in words:
            processed_words.append(word)
            if word in punctuations:
                processed_words.append('<eos>')
        processed_words.append('<eos>')  # Append <eos> at the end of the line
        return processed_words
