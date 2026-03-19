from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

for X_batch,Y_batch in test_dataloader:
    print(f'shape of X_batch :{X_batch.shape}')
    print(f'shape of Y_batch :{Y_batch.shape}')
    break