
import load_data
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as CrossEntropyLoss
from torch.autograd import grad
import torch.optim as optim
from Classifier import LSTMClassifier
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters, feel free to tune


batch_size = 27
output_size = 9   # number of class
hidden_size = 50  # LSTM output size of each time step
input_size = 12
basic_epoch = 1
Adv_epoch = 50
Prox_epoch = 50

def plot(history,  figType = 'acc', saveFile = None, mode = 'plain'):
    if figType == 'acc':
        fig = plt.figure()
        plt.title('Test Accuracy vs Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        xAxis = np.arange(len(history[1]))+1
        if mode == 'plain':
            plt.plot(xAxis, history[1], marker = 'o', label = 'plain LSTM Acc')
        elif mode == 'adv':
            for i in range(len(history)):
                xAxis = np.arange(len(history[i]))+1
                plt.plot(xAxis, history[i], marker = 'o', label = 'e= '+str(self.epsilons[i]))
        plt.legend()
        if saveFile is not None:
            plt.savefig(saveFile)
        return fig



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)



# Training model
def train_model(model, train_iter, mode, epsilon= 0.1):
    total_epoch_loss = 0
    total_epoch_acc = 0
    steps = 0
    model.train()
    model2 = model
    for idx, batch in enumerate(train_iter):
        input = batch[0]
        target = batch[1]
        target = torch.autograd.Variable(target).long()
        r = 0
        optim.zero_grad()
        prediction = model2(input, r,batch_size = input.size()[0], mode = mode)
        #print(target, prediction)
        #print(target.shape, prediction.shape)
        loss = loss_fn(prediction, target)
        if mode == 'AdvLSTM':
            pert = compute_perturbation(loss, model)
            ''' Add adversarial training term to loss'''
            r =  epsilon * pert
            prediction = model2(input, r,batch_size = input.size()[0], mode = 'AdvLSTM')

        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/(input.size()[0])
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


# Test model
def eval_model(model, test_iter, mode):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    r = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            input = batch[0]
            target = batch[1]
            target = torch.autograd.Variable(target).long()
            prediction = model(input, r, batch_size=input.size()[0], mode = mode)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects.double()/(input.size()[0])
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(test_iter), total_epoch_acc / len(test_iter)




def compute_perturbation(loss, model):
    gradDir = grad(loss, model.inX, retain_graph = True)[0] # This returns a tuple
    magnitude = gradDir.pow(2).sum().sqrt()
    '''need to be implemented'''

    return gradDir/magnitude #the value of g / ||g||



''' Training basic model '''

train_iter, test_iter = load_data.load_data('JV_data.mat', batch_size)


model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
#loss_fn = F.cross_entropy
loss_fn = CrossEntropyLoss()

for epoch in range(basic_epoch):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)
        train_loss, train_acc = train_model(model, train_iter, mode = 'plain')
        model.history[0].append(train_acc)
        val_loss, val_acc = eval_model(model, test_iter, mode ='plain')
        model.history[1].append(val_acc)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')

model.plot(saveFile = './plots/LSTM_plain_acc.png')

''' Save and Load model'''

# 1. Save the trained model from the basic LSTM
model.save()
# 2. load the saved model to Prox_model, which is an instance of LSTMClassifier
    #Prox_model = ..., or other implementations

# 3. load the saved model to Adv_model, which is an instance of LSTMClassifier

epsilons = [0.01,0.1,1]
histories = [[],[],[]]
for i, e in enumerate(epsilons):
    Adv_model = LSTMClassifier(batch_size, output_size, hidden_size, input_size)
    Adv_model.load()
    for epoch in range(Adv_epoch):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Adv_model.parameters()), lr=5e-4, weight_decay=1e-4)
        train_loss, train_acc = train_model(Adv_model, train_iter, mode = 'AdvLSTM', epsilon = e)
        Adv_model.history[0].append(train_acc)
        val_loss, val_acc = eval_model(Adv_model, test_iter, mode ='AdvLSTM')
        Adv_model.history[1].append(val_acc)
        histories[i].append(val_acc)
        print(f'Epsilon: {e}, Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')
    model.plot(saveFile = './plots/LSTM_adv_epsilon_'+str(e)+'_acc.png', label = 'Adv LSTM e: ' +str(e))
plot(histories, saveFile = './plots/LSTM_Adv_acc_all.png', mode = 'adv')


''' Training Prox_model'''
for epoch in range(Prox_epoch):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, Prox_model.parameters()), lr=1e-3, weight_decay=1e-3)
    train_loss, train_acc = train_model(Prox_model, train_iter, mode = 'ProxLSTM')
    val_loss, val_acc = eval_model(Prox_model, test_iter, mode ='ProxLSTM')
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Test Loss: {val_loss:3f}, Test Acc: {val_acc:.2f}%')






