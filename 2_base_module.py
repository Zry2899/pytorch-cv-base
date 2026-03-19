import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64

train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=False,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)

train_dataloader = DataLoader(train_data,batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data,batch_size=BATCH_SIZE)

class Base_NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28*28,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=10),
        )
    def forward(self,X):
        X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        return logits
    
model = Base_NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(),lr=1e-3)

def train(model,dataloader,loss_function,opt):
    size = len(dataloader.dataset)
    model.train()
    for batch,(X,Y) in enumerate(dataloader):
        X,Y = X.to(device),Y.to(device)

        pred = model(X)
        loss = loss_function(pred,Y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch % 100 ==0:
            loss,current = loss.item(),(batch+1)*len(X)
            print(f"loss:{loss:>7f}   [{current:>5d}/{size:>5d}]")

def test(model,dataloader, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model,train_dataloader,loss_fn,opt)
    test(model,train_dataloader,loss_fn)
print("Done!")

torch.save(model.state_dict(),"model.pth")
print("We saved PtTorch Model State to model.pth successful")