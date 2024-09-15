import numpy as np


class DictList:
    def __init__(self):
        self.dico = {}

    def __setitem__(self, key, value):
        if key in self.dico:
            self.dico[key].append(value)
        else:
            self.dico[key] = [value]

    def getmean(self):
        dico = {}
        for key, val in self.dico.items():
            dico[key] = np.concatenate(val).mean()
        return dico