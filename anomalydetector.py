import numpy as np
from collections import deque


class AnomalyDetector:
    def __init__(self, window=50, threshold=3.5):
        self.losses = deque(maxlen=window)
        self.entropies = deque(maxlen=window)
        self.threshold = threshold

    def is_anomalous(self, loss, entropy):
        self.losses.append(loss)
        self.entropies.append(entropy)

        if len(self.losses) < 10:
            return False

        lossz = (loss - np.mean(self.losses)) / (np.std(self.losses) + 1e-8)
        entz = (entropy - np.mean(self.entropies)) / (np.std(self.entropies) + 1e-8)

        return abs(lossz) > self.threshold or abs(entz) > self.threshold
