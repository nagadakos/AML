from conv import Conv
import torch

input_X = torch.tensor([[[[1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1],
                        [0, 0, 1, 1, 0],
                        [0, 1, 1, 0, 0]]]], dtype=torch.float)

# input must be four dimensional

input_K = torch.tensor([[[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]]], dtype=torch.float)
# kernel must be three dimensional: num_filters x rows x columns


cl = Conv(3, 1)
cl.init_params(kernel=input_K, bias=torch.tensor([0], dtype=torch.float))

output = cl(input_X)
print(output)


