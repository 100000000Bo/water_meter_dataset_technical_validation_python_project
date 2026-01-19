import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  # If the output is True, the installation is successful