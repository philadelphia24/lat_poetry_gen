import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

class Dictionary(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.word2idx = tokenizer.word_index  # Keras tokenizer word_index
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}  # Reverse mapping

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx) + 1
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self.word2idx.get(word, None)

    def __len__(self):
        return len(self.word2idx)

class Corpus(object):
    def __init__(self, path, tokenizer):
        self.dictionary = Dictionary(tokenizer)
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file using the Keras tokenizer."""
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            tokens = []
            for line in f:
                words = line.split()
                for word in words:
                    if word not in self.dictionary.word2idx:
                        self.dictionary.add_word(word)
                    tokens.append(self.dictionary.word2idx.get(word, 0))
        return tokens

def load_tokenizer(tokenizer_path):
    """Load the Keras tokenizer from a pickle file."""
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

if __name__ == "__main__":
    tokenizer_path = '/home/pricie/cclstudent10/latin_poetry/ten_million/vocab_10k/tokenizer_train_10mio_with_EOS.pickle'
    data_path = '/home/pricie/cclstudent10/pytorch_model/ophelia/examples/word_language_model/data/corpus_10mio_10k'

    tokenizer = load_tokenizer(tokenizer_path)
    print("Tokenizer loaded:", tokenizer)
    print("Data path:", data_path)
    corpus = Corpus(data_path, tokenizer)  # Correctly pass both arguments
    print("Corpus created successfully.")
