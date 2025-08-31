import torch
print(torch.__version__)

tensor0d = torch.tensor(0)

tensor1d = torch.tensor([1, 2, 3])

tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

tensor4d = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])

print(tensor3d.shape)