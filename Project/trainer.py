import torch.nn.functional as F
from indexes import CIDX as cidx
import torch
import ipdb




def train_classifier(model, indata, device, lossFn, optim,  **kwargs):
    
    verbose = True if 'verbose' not in kwargs.keys() else kwargs['verbose']
    epoch = kwargs['trainerArgs']['epoch'] if 'trainerArgs' in kwargs.keys() else -1
    stopIdx = kwargs['trainerArgs']['stopIdx'] if 'trainerArgs' in kwargs.keys() else 0 #used for early stopping at the target batch number
    printInterval = kwargs['trainerArgs']['printInterval'] if 'trainerArgs' in kwargs.keys() else 40
    factor = 0
    totalSize, totalLoss = 0, 0
    print("Train Device is {}".format(device))
    for idx, items in enumerate(indata):
        
        # A. Forward computation input retrieval and handling
        if type(items) == list:
            data   = items[0]
            target = items[1]
        else:
            target = items['target'].float() if any(items['target']) else None
            data = items['data']
            
        if not type(data) in (tuple, list):
            data = (data,)
        data = tuple(d.to(device) for d in data)
        
        if target is not None:
            target = target.to(device)
                
        # B. Forward pass calculate output of model
        output      = model.encoder.forward(*data)
                    
        # C. Loss computation part.
        # Convention for all loss and reconstruction inputs is Data, Target, miscInputs. Model forward MUST be
        # Designed to match its output to the loss functions' input pattern.
        if type(output) not in (tuple, list):
            output = (output,)
        # 1 position: data
        lossInputs = (output[0],)
        if target is not None:
            target = (target,)
        # 2 position target
            lossInputs += target
        # 3: positions-> rest of required misc Inputs to loss func.
        lossInputs += tuple(output[1:])
        
        # compute loss
        #ipdb.set_trace() # BREAKPOINT
        lossOutputs = lossFn(*lossInputs)
        loss  = lossOutputs[0] if type(lossOutputs) in (tuple, list) else lossOutputs
        #loss        = lossFn(output, label)
        totalLoss  += loss
        # loss        = F.CrossEntropyLoss(output, target)

        # D. Backpropagation part
        # 1. Zero out Grads
        optim.zero_grad()
        # 2. Perform the backpropagation based on loss
        loss.backward()            
        # 3. Update weights 
        optim.step()

       # E. Training Progress report for sanity purposes! 
        if verbose:
            if idx % printInterval == 0:
                    print("Epoch: {}, Batch: {} / {} ({:.0f}%). Loss: {:.4f}".format(epoch, idx,len(indata),100.*idx/len(indata), loss.item()))
                
        if stopIdx and idx == stopIdx:
                print("Stop index reached ({}). Stopping training". format(stopIdx))
                break
        
    totalSize += len(indata.dataset)
    # --|
    # F. Logging part
    avgLoss  = totalLoss  / totalSize
    model.metric = avgLoss
    # Log the current train loss
    model.history[cidx.trainLoss].append(avgLoss)
    model.history[cidx.trainAcc].append(0)
    model.history[cidx.trainAcc5].append(0)
    return loss, 1

# -----------------------------------------------------------------------------------------------------------------------

# Testing and error reports are done here
def test_classifier(model, testLoader, device, lossFn, **kwargs):
    """ DESCRIPTION: This function handles testing performance of a model. It can be modified to accept a varying number of inputs.
        
        RETURNS: acc (float): The reported average top-1 accuracy for this trial.
                 loss (float): The reported average loss on the test data, for this trial.
    """
    if 'testerArgs' in kwargs.keys():
        earlyStopIdx = kwargs['testerArgs']['earlyStopIdx'] if 'earlyStopIdx' in kwargs['testerArgs'].keys() else 0
            
    print("In Testing Function!")        
    loss = 0 
    true = 0
    acc  = 0
    # Inform Pytorch that keeping track of gradients is not required in
    # testing phase.
    with torch.no_grad():
        for idx, (data, label) in enumerate(testLoader):
            data, label = data.to(device), label.to(device)
            # output = self.forward(data)
            output = model.encoder.forward(data)
            # Sum all loss terms and tern then into a numpy number for late use.
            loss  += lossFn(output, label).item()
            # Find the max along a row but maitain the original dimenions.
            # in this case  a 10 -dimensional array.
            pred   = output.max(dim = 1, keepdim = True)
            # Select the indexes of the prediction maxes.
            # Reshape the output vector in the same form of the label one, so they 
            # can be compared directly; from batchsize x 10 to batchsize. Compare
            # predictions with label;  1 indicates equality. Sum the correct ones
            # and turn them to numpy number. In this case the idx of the maximum 
            # prediciton coincides with the label as we are predicting numbers 0-9.
            # So the indx of the max output of the network is essentially the predicted
            # label (number).
            true  += label.eq(pred[1].view_as(label)).sum().item()
            
            if earlyStopIdx and idx == earlyStopIdx:
                print("Stop index reached ({}). Stopping training". format(earlyStopIdx))
                break
                
    acc = true/len(testLoader.dataset)
    model.history[cidx.testAcc].append(acc)
    model.history[cidx.testAcc5].append(acc) 
    model.history[cidx.testLoss].append(loss) 
    # Print accuracy report!
    print("Accuracy: {} ({} / {})".format(acc, true,
                                          len(testLoader.dataset)))
    return acc, loss

# -----------------------------------------------------------------------------------------------------------------------

def dynamic_conv_check(lossHistory, args = dict(window = 2, percent_change = 0.01, counter = 0, lossIdx =
                                                cidx.testLoss )):
    ''' Description: This function checks whether train should stop according to progress made.
                     If the criteria is met, training will halt.
        Arguments:   lossHistory (list): a list of lists containing loss and MAE, MAPE, as
                                         indexex by ridx file.
                     args:  A dictionary with operational parameters.
                     w:     Length of history to tbe considered
    '''
    w = args['window']
    perc = args['percent_change']
    counter = args['counter']
    lossIdx = args['lossIdx']
    # Sanitazation
    # check for parameter validity
    w = w if len(lossHistory[lossIdx]) >= w else len(lossHistory[lossIdx])
    # Increase the watch counter. When the track counter surpasses the target window length
    # check to see if progress  has been made. Essentially, by setting counter to 0 the function
    # will wait for window checks, again, before checking for convergence.
    counter += 1
    if counter >= w:
        # Compute average loss difference from history
        loss_i_1 = lossHistory[lossIdx][-w]
        avg_loss = loss_i_1
        loss_diff= 0
        for i in lossHistory[lossIdx][-w+1:]:
            loss_diff += loss_i_1 - i
            avg_loss += i
            loss_i_1 = i
        loss_diff /=w
        avg_loss /= w
        target = avg_loss * perc
        # if average loss change is less than a percentage of the average loss, exit.
        if loss_diff  < target and len(lossHistory[lossIdx]) > args['window']:
            print('!!!!!!!!!!!!!!!!!!!!')
            print("Progress over last {} epochs is less than {}% ({:.4f}<{:.4f}). Exiting Training".format(w,perc*100, loss_diff, target))
            print('!!!!!!!!!!!!!!!!!!!!')
            return 1
        else:
            return 0

    else:
        return 0