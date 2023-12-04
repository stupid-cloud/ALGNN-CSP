import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from typing import Optional
import numpy as np

# Prints model summary
def model_summary(model):
    model_params_list = list(model.named_parameters())
    print("--------------------------------------------------------------------------")
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("--------------------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("--------------------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)


def global_first_pool(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    batch_size = torch.unique(batch).shape[0]
    index_list = []
    for i in range(batch_size):
        index = batch.detach().tolist().index(i)
        index_list.append(torch.IntTensor([index]).cuda())
    summed_fea = [torch.index_select(x, 0, index) for index in index_list]
    return torch.cat(summed_fea, dim=0)
