import torch
import numpy as np

advice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = np.array([[1,2],
        [3,4]])

x_tensor = torch.tensor(data) # this method will copy the data from data
x_tensor ==torch.from_numpy(data) # this method share the same memory with data

rand_tensor = torch.rand((2,3))
ones_tensor = torch.ones((2,3))
zeros_tensor = torch.zeros((2,3))

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
print(f"Zeros Tensor's shape: \n {zeros_tensor.shape}")
print(f"Zeros Tensor's dtype: \n {zeros_tensor.dtype}")
print(f"Zeros Tensor's device: \n {zeros_tensor.device}")

zeros_tensor = zeros_tensor.to(advice)

print(f"Zeros Tensor's device: \n {zeros_tensor.device}")

x_cat_tensor = torch.cat([rand_tensor,rand_tensor],dim=1) # dim=1 means you cat the two tensor in dimention 1,1 means the second number in shape
