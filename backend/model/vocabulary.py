from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"} # index-to-string mapping (used to decode).
        self.stoi = {v: k for k, v in self.itos.items()} # string-to-index mapping (used to encode).

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in sentence.lower().split():
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])