from torch.serialization import check_module_version_greater_or_equal
import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from torch.nn.utils.rnn import pad_sequence



class DynaConv2d(_ConvNd):


    def __init__(
        self,
        in_channels,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)

        super(DynaConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)



    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor, action :Tensor,device="cuda") -> Tensor:        
        bt = input.shape[0]
        H,W = self.calc_shape(input.shape[2],input.shape[3])
        output = torch.empty((bt,self.out_channels,H,W),device="cuda")
        for i in range(bt):
            in_bt = torch.split(input[i],action[i],dim=0)[0].unsqueeze(0)
            #print(ch_in, in_bt.shape[1])

            gated_weight = torch.split(self.weight,action[i],dim=1)[0]
            tmp = self._conv_forward(in_bt,gated_weight,self.bias)
            output[i] = tmp
        
        return output  
    

    def calc_shape(self,H,W): 
        H_prime = (H - self.kernel_size[0]+2*self.padding[0])/self.stride[0]  +1 
        W_prime = (W - self.kernel_size[0]+2*self.padding[0])/self.stride[0] +1
        return int(H_prime), int(W_prime)

                
                

        