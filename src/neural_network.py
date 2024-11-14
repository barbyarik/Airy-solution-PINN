'''The neural_network.py file provides interaction with the neural network, \
its training and use'''

from tqdm import tqdm
import torch, torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data_numerical import y_0, y_prime_0
from sklearn.metrics import mean_absolute_percentage_error as mape

def _choose_device():
    '''Search for the best available computing device'''

    print(f"Is CUDA supported by this system?")
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print("No, the CPU will be used\n")
    else:
        device = torch.device('cuda')
        print("Yes, the GPU will be used")
        print(f"CUDA version: {torch.version.cuda}")
        cuda_id = torch.cuda.current_device()
        print(f"ID of the current CUDA device: {torch.cuda.current_device()}")
        print(f"The name of the current CUDA device: {torch.cuda.get_device_name(cuda_id)}\n")
    
    return device

class Sin(nn.Module):
    '''Implements the sin activation function'''

    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class MeanSquaredErrorPINN(nn.Module):
    '''A custom error function that takes into account the equation \
as one of the parts of the error'''

    def __init__(self, lamb):
        super(MeanSquaredErrorPINN, self).__init__()
        self.lamb = lamb

    def forward(self, y_nn, x, model):
        y_prime = torch.autograd.grad(y_nn, x, grad_outputs=torch.ones_like(y_nn),
                                      create_graph=True)[0]
        y_second_prime = torch.autograd.grad(y_prime, x, grad_outputs=torch.ones_like(y_nn),
                                             create_graph=True)[0]
        
        equation_loss = torch.mean((y_second_prime - y_nn * x) ** 2)

        x_0 = torch.tensor([0.], requires_grad=True)
        y_nn_0 = model(x_0)
        y_nn_prime_0 = torch.autograd.grad(y_nn_0, x_0, create_graph=True)[0]

        boundary_loss = (y_nn_0 - y_0) ** 2 + (y_nn_prime_0 - y_prime_0) ** 2

        return equation_loss + self.lamb * boundary_loss

def _compile_model(layers_list, device):
    '''Creates a fully connected neural network and \
transfers it to the selected device'''

    return nn.Sequential(*layers_list).to(device=device)

def _fit(model, optimizer, epochs, device, criterion, train_loader, gamma):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)
    for epoch in range(1, epochs+1):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, (x, y_numerical) in loop:
            x = x.to(torch.float32).to(device=device)
            y_numerical = y_numerical.to(torch.float32).to(device=device)
            x = x.reshape(x.shape[0], -1)
            x.requires_grad=True
            y_numerical = y_numerical.reshape(y_numerical.shape[0], -1)
            y_nn = model(x)
            loss = criterion(y_nn, x, model)
            acc = 1 - mape([t[0] for t in y_numerical.tolist()],
                           [s[0] for s in y_nn.tolist()])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(loss=loss.item(), acc=acc, lr=scheduler.get_last_lr()[0])
        scheduler.step()

def save_checkpoint(checkpoint, filename='new_checkpoint'):
    '''Saves the checkpoint in tar format'''

    print('--> The checkpoint of the model has been saved. \
The result can be seen in data/checkpoints/\n')
    torch.save(checkpoint, f'data/checkpoints/{filename}.pth.tar') 

def load_checkpoint(filename):
    '''Loads a checkpoint from the tar format'''

    print('--> The checkpoint of the model has been loaded.\n')
    return torch.load(f'data/checkpoints/{filename}.pth.tar')

def launch(model, train_data, epochs=250, batchsize=128, lr=0.005, lamb=1.0, gamma=0.95):
    '''An interface function that provides training/retraining of \
the model according to the specified parameters'''

    device = _choose_device()
    x_list = list(train_data['x_value'].values)
    y_numerical = list(train_data['y_value'].values)
    train_loader = DataLoader([(x, y) for x, y in train_data.values],
                              batch_size=batchsize, shuffle=True)
    if isinstance(model, list): model = _compile_model(model, device)
    else: model = model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MeanSquaredErrorPINN(lamb)

    _fit(model, optimizer, epochs, device, criterion, train_loader, gamma)
    acc = 1 - mape(y_numerical, [float(model(torch.Tensor([x]))) for x in x_list])
    checkpoint = {'model': model, 'acc': acc}
    save_checkpoint(checkpoint, filename=f'chekpoint_{round(acc, 3)}_acc')
  
    return model