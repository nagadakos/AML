import gzip
import csv
import numpy as np
import torch

class DataLoader:
    def __init__(self, type= 'word-features'):
        data_path = '../data/letter.data.gz'
        lines = self._read(data_path)
        if type == 'word-features':
            data, target = self._parse(lines)
            #self.data, self.target = self._parse(lines)
            self.data, self.target = self._pad(data, target)
        else:
            #print(lines[0:20])
            self.data, self.target = self._letter_parse(lines)

    @staticmethod
    def _read(filepath):
        with gzip.open(filepath, 'rt') as file_:
            reader = csv.reader(file_, delimiter='\t')
            lines = list(reader)
            return lines

    @staticmethod
    def _parse(lines):
        lines = sorted(lines, key=lambda x: int(x[0]))
        data, target = [], []
        next_ = None

        for line in lines:
            if not next_:
                data.append([])
                target.append([])
            else:
                assert next_ == int(line[0])
            next_ = int(line[2]) if int(line[2]) > -1 else None
            pixels = np.array([int(x) for x in line[6:134]])
            pixels = pixels.reshape((16, 8))
            data[-1].append(pixels)
            target[-1].append(line[1])
        return data, target
    
    @staticmethod
    def _letter_parse(lines):
        lines = sorted(lines, key=lambda x: int(x[0]))
        data, target = [], []
        data   = np.zeros((len(lines),16,8))
        target = np.zeros((len(lines),1))
        next_ = None

        for i,line in enumerate(lines):
            pixels = np.array([int(x) for x in line[6:134]])
            pixels = pixels.reshape((16, 8))
            data[i] = pixels
            target[i] = ord(line[1]) - ord('a')
        # One-hot encode targets.
        #target1Hot = np.zeros((len(target),26))
        #for index, letter in enumerate(target):
        #    if letter:
        #        target1Hot[index][ord(letter) - ord('a')] = 1
        return data, target


    @staticmethod
    def _pad(data, target):
        """
        Add padding to ensure word length is consistent
        """
        max_length = max(len(x) for x in target)
        padding = np.zeros((16, 8))
        data = [x + ([padding] * (max_length - len(x))) for x in data]
        target = [x + ([''] * (max_length - len(x))) for x in target]
        return np.array(data), np.array(target)

def get_dataset(type = 'word-features', convToTensor = False):
    
    if type == 'word-features':
        dataset = DataLoader(type=type)

        print(dataset.data[0].shape)
        # Flatten images into vectors.
        #dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))
        dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))

         # One-hot encode targets.
        target = np.zeros(dataset.target.shape + (26,))
        for index, letter in np.ndenumerate(dataset.target):
            if letter:
                target[index][ord(letter) - ord('a')] = 1
        dataset.target = target

        
    else:
        print("Loading Dataset as Letter-features for Deepnet comparison.\n")
        dataset = DataLoader(type=type)
        dataset.data = np.reshape(dataset.data,(dataset.data.shape[0],1,dataset.data.shape[1], dataset.data.shape[2]))
    if convToTensor:
        # Convert dataset into torch tensors
        dataset.data = torch.tensor(dataset.data).float()
        dataset.target = torch.tensor(dataset.target).long()
        
    # Shuffle order of examples.
    order = np.random.permutation(len(dataset.data))
    dataset.data = dataset.data[order]
    dataset.target = dataset.target[order]
    return dataset
