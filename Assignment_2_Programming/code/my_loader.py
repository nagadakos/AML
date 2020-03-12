import gzip
import csv
import numpy as np
import torch
import torch.utils.data as data_utils

class DataLoader:
    def __init__(self, dType= 'word-features'):
        data_path = '../data/all_data.txt'
        lines = self._read(data_path)
        self.wordIdxs = 0
        if dType == 'word-features':
            data, target = self._parse(lines)
            self.data, self.target = self._pad(data, target)
        else:
            self.data, self.target = self._parse2(lines)
            #self.data, self.target, self.wordIdxs = self._letter_parse(lines)

    @staticmethod
    def _read(filepath):
        with open(filepath, 'rt') as f:
            lines = [[y for y in x.split(' ')] for x in f.read().splitlines()]
            #reader = csv.reader(file_, delimiter='\t')
            #lines = list(reader)
            return lines

    @staticmethod
    def _parse(lines):
        """ DESCRIPTION: Load data for word-wise purposes.
        """
        #lines = sorted(lines, key=lambda x: int(x[0]))
        data, target = [], []
        next_ = None
        i = 0
        for line in lines:
            if not next_:
                data.append([])
                target.append([])
            next_ = int(line[2]) if int(line[2]) > -1 else None
            pixels = np.array([int(x) for x in line[5:]])
            pixels = pixels.reshape((16, 8))
            data[-1].append(pixels)
            target[-1].append(line[1])
        return data, target
    
    @staticmethod
    def _parse2(lines):
        """ DESCRIPTION: Load data for letter-wise purposes.
        """
        data, target, wordIdxs = [], [], []
        next_ = None
        i = 0
        for line in lines:
            if not next_:
                data.append([])
                target.append([])
                wordIdxs.append([])
            next_ = int(line[2]) if int(line[2]) > -1 else None
            pixels = np.array([int(x) for x in line[5:]])
            pixels = pixels.reshape((16, 8))
            data[-1].append(pixels)
            target[-1].append(ord(line[1]) - ord('a'))
            wordIdxs[-1].append(int(line[0])-1)
        return data, target

    
    @staticmethod
    def _letter_parse(lines):
        lines = sorted(lines, key=lambda x: int(x[0]))
        data, target = [], []
        data   = np.zeros((len(lines),16,8))
        target = np.zeros((len(lines),1))
        wordIdxs   = np.zeros((len(lines),1))
        next_ = None

        for i,line in enumerate(lines):
            pixels = np.array([int(x) for x in line[5:]])
            pixels = pixels.reshape((16, 8))
            data[i] = pixels
            target[i] = ord(line[1]) - ord('a')
            letterIdxs[i] = line[3]
        # One-hot encode targets.
        #target1Hot = np.zeros((len(target),26))
        #for index, letter in enumerate(target):
        #    if letter:
        #        target1Hot[index][ord(letter) - ord('a')] = 1
        return data, target, letterIdxs
    
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
    
# -------------------------------------------------------------------------------------------------------------
def words_to_letters(data, target):
    """ Description: THis funciton will conver a list of words to a numpy array of letters.
    """
    #data = [w for w in data]
    # [item for sublist in l for item in sublist]
    data = [l for w in data for l in w]
    wordLimits = np.zeros((len(target),1))
    for i, t in enumerate(target[1:]):
        wordLimits[i+1] = wordLimits[i] + len(t)
    target = [l for w in target for l in w]
    return np.array(data), np.array(target), np.array(wordLimits)

# -----------------------------------------------------------------

def load_data(bSize = 32, dType = 'word-features', split = 0.5):
    # bundle common args to the Dataloader module as a kewword list.
    # pin_memory reserves memory to act as a buffer for cuda memcopy 
    # operations
    comArgs = {'shuffle': True,'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    trainWordLimits, testWordLimits = 0, 0
    # Data Loading -----------------------
    # ******************
    # load the data set as a colldection of tuples. First element in tuple are the data formated as a numOFLetters / word x 128; second 
    # elements is a one-hot endoding of the label of the letter so: numOfLetter / word x 26. Zero padding has been used to keep word size
    # consistent to 14 letters per word. The labels for the padding rows are all 0s.
    # ******************
    dataset = get_dataset(dType=dType, convToTensor = False)
    split = int(split * len(dataset.data)) # train-test split
    train_data, test_data = dataset.data[:split], dataset.data[split:]
    train_target, test_target = dataset.target[:split], dataset.target[split:]
    
    if dType == 'letter-features':
        #trainWordIdxs, testWordIdxs = dataset.wordIdxs[:split], dataset.wordIdxs[split:]
        train_data, train_target, trainWordLimits = words_to_letters(train_data, train_target)
        test_data, test_target, testWordLimits =  words_to_letters(test_data, test_target)
        train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1], train_data.shape[2]))
        test_data  = np.reshape(test_data,  (test_data.shape[0],  1, test_data.shape[1],  test_data.shape[2]))
    # Convert dataset into torch tensors
    train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
    test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())
    # Define train and test loaders
    trainLoader = data_utils.DataLoader(train,  # dataset to load from
                                         batch_size=bSize,  # examples per batch (default: 1)
                                         shuffle=True,
                                         sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                         num_workers=5,  # subprocesses to use for sampling
                                         pin_memory=False,  # whether to return an item pinned to GPU
                                         )

    testLoader = data_utils.DataLoader(test,  # dataset to load from
                                        batch_size=bSize,  # examples per batch (default: 1)
                                        shuffle=False,
                                        sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                        num_workers=5,  # subprocesses to use for sampling
                                        pin_memory=False,  # whether to return an item pinned to GPU
                                        )
    print('Loaded dataset... ')
    # End of DataLoading -------------------


    # Sanity Prints---
    #print(testWordLimits[0:200])
    #print(len(train))
    #print(type(train[0]))
    #print(train[0][0].shape)
    #print(train[0][1].shape)
    #print(train[0][1].shape)
    return trainLoader, testLoader, [train_target, test_target, trainWordLimits, testWordLimits, train_data, test_data]

# -------------------------------------------------------------------------------------------------------------
def get_dataset(dType = 'word-features', convToTensor = False):
    
    if dType == 'word-features':
        dataset = DataLoader(dType=dType)
        
        # Flatten images into vectors.
        #dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))
        dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))

         # One-hot encode targets.
        target = np.zeros(dataset.target.shape + (26,))
        for index, letter in np.ndenumerate(dataset.target):
            if letter:
                target[index][ord(letter) - ord('a')] = 1
        dataset.target = target
        # Shuffle order of examples.
        order = np.random.permutation(len(dataset.data))
        dataset.data = dataset.data[order]
        dataset.target = dataset.target[order]
    else:
        print("Loading Dataset as Letter-features for Deepnet comparison.\n")
        dataset = DataLoader(dType=dType)
        #dataset.data = np.reshape(dataset.data,(dataset.data.shape[0],1,dataset.data.shape[1], dataset.data.shape[2]))
    if convToTensor:
        # Convert dataset into torch tensors
        dataset.data = torch.tensor(dataset.data).float()
        dataset.target = torch.tensor(dataset.target).long()
        
    
    return dataset


def main():
    #get_dataset(dType = 'letter', convToTensor = False)
    load_data(dType= 'letter-features')
    
    
if __name__ == '__main__':
    main()