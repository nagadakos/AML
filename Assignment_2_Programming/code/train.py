import torch
import torch.optim as optim
import torch.utils.data as data_utils
from data_loader import get_dataset
import numpy as np
from crf import CRF_NET


# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Tunable parameters
batch_size = 256
num_epochs = 10
max_iters  = 1000
print_iter = 25 # Prints results every n iterations
conv_shapes = [[1,64,128]] #


# Model parameters
input_dim = 128
embed_dim = 64
num_labels = 26
cuda = torch.cuda.is_available()

# Instantiate the CRF model
#crf = CRF(input_dim, embed_dim, conv_shapes, num_labels, batch_size)
crf = CRF_NET((16,8), padding = True)
# Setup the optimizer
opt = optim.LBFGS(crf.parameters())




def evaluate_crf_predictions(pred, letterLabels):
    letterAcc, wordAcc = 0.0,0.0
    
    for i, label in enumerate(letterLabels):       
        
        # Get correct labels
        decSeq = pred[i]
        
        res = torch.sum((decSeq == label).to('cuda'))     # find the number of labels that are equal to ground Truth
        letterAcc += res                                                              # Letterwise acc is increases for every match found
        wordAcc += 1 if res == decSeq.shape[0] else 0                                 # word acc increases only when ALL labels are correct
    # Average out letter-wise acc over all letter and word-wise over all words.  
    letterAcc /= pred.numel()
    wordAcc /= pred.shape[0] # len gives as the number of words (each word is a line and the value is at which letter index it ends!)
    return letterAcc, wordAcc


##################################################
# Begin training
##################################################
step = 0

# Fetch dataset
dataset = get_dataset()
split = int(0.5 * len(dataset.data)) # train-test split
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Convert dataset into torch tensors
train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())

# Define train and test loaders
train_loader = data_utils.DataLoader(train,  # dataset to load from
                                     batch_size=batch_size,  # examples per batch (default: 1)
                                     shuffle=True,
                                     sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                     num_workers=0,  # subprocesses to use for sampling
                                     pin_memory=False,  # whether to return an item pinned to GPU
                                     )

test_loader = data_utils.DataLoader(test,  # dataset to load from
                                    batch_size=batch_size,  # examples per batch (default: 1)
                                    shuffle=False,
                                    sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                    num_workers=0,  # subprocesses to use for sampling
                                    pin_memory=False,  # whether to return an item pinned to GPU
                                    )
print('Loaded dataset... ')
for i in range(num_epochs):
    print("Processing epoch {}".format(i))
    # Now start training
    if False:
        for i_batch, sample in enumerate(train_loader):

            train_X = sample[0]
            train_Y = sample[1]

            if cuda:
                train_X = train_X.cuda()
                train_Y = train_Y.cuda()

            # compute loss, grads, updates:
            def closure():
                opt.zero_grad()
                _ = crf.forward([train_X, train_Y])
                tr_loss = crf.loss(train_X, train_Y) # Obtain the loss for the optimizer to minimize
                tr_loss.backward() # Run backward pass and accumulate gradients
                return tr_loss

            tr_loss = crf.loss(train_X, train_Y)
            #opt.zero_grad() # clear the gradients
            #_ = crf.forward(sample)
            #tr_loss = crf.loss(train_X, train_Y) # Obtain the loss for the optimizer to minimize
            #tr_loss.backward() # Run backward pass and accumulate gradients
            opt.step(closure) # Perform optimization step (weight updates)

            # print to stdout occasionally:
            if step % print_iter == 0:
                random_ixs = np.random.choice(test_data.shape[0], batch_size, replace=False)
                test_X = test_data[random_ixs, :]
                test_Y = test_target[random_ixs, :]

                # Convert to torch
                test_X = torch.from_numpy(test_X).float()
                test_Y = torch.from_numpy(test_Y).long()

                if cuda:
                    test_X = test_X.cuda()
                    test_Y = test_Y.cuda()
                test_loss = crf.loss(test_X, test_Y)
                tr_loss = tr_loss.item() if 'Tensor' in str(type(tr_loss)) else tr_loss
                print(step, tr_loss, test_loss.item(),
                           tr_loss / batch_size, test_loss.item() / batch_size)

##################################################################
# IMPLEMENT WORD-WISE AND LETTER-WISE ACCURACY HERE
##################################################################
    for t_batch, sample in enumerate(test_loader): 
        if cuda:
            sample = [s.cuda() for s in sample]
            
        preds = crf.predict(sample)
        labels = sample[1]
        lAcc, wAcc = evaluate_crf_predictions(preds, labels)
        print("Letter Accuracy: {}, Word Accuracy: {}".format(lAcc, wAcc))
        
        step += 1
        if step > max_iters: raise StopIteration
    del train, test
