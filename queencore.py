import numpy as np


class QueenAuoraBigram:
    def __init__(self, vocabsize=256, dim=128):
        self.emb = np.random.randn(vocabsize, dim) * 0.1
        self.Wh = np.random.randn(dim, dim) * 0.1
        self.Wout = np.random.randn(dim, vocabsize) * 0.1

    def forward(self, xbytes):
        x = np.array(xbytes, dtype=np.uint8).reshape(1, -1)
        h = np.tanh(self.emb[x].sum(axis=1))
        logits = h @ self.Wout
        return logits, h
