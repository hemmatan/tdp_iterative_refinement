class Vocabulary:

    def __init__(self, symbols=None):
        self.symb2idx = {}
        self.idx2sym = []
        if symbols:
            self.update(symbols)

    def __len__(self):
        return len(self.idx2sym)

    def update(self, symbol_list):
        for S in symbol_list:
            if S not in self.symb2idx:
                self.symb2idx[S] = len(self.idx2sym)
                self.idx2sym.append(S)

    def from_dictionary(self, vocab):
        self.idx2sym = [-1 for i in range(len(vocab))]
        for token in vocab.keys():
            self.symb2idx[token] = vocab[token]
            self.idx2sym[vocab[token]] = token

    def rev_lookup(self, idx):
        return self.idx2sym[idx]

    def __call__(self, symbol, fallback=None):
        return self.lookup(symbol, fallback)

    def lookup(self, symbol, fallback=None):
        if fallback:
            return self.symb2idx.get(symbol, self.symb2idx[fallback])
        else:
            return self.symb2idx[symbol]
