
import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    Convolution layer.
    """
    def __init__(self, kernel_size, out_channels, padding=False, stride=(1, 1)):
        super(Conv, self).__init__()
        # start kernel with random init
        self.kernel_size = kernel_size
        # set number of filters
        self.num_filters = out_channels
        assert self.num_filters >= 1

        self.bias = torch.rand(self.num_filters, requires_grad=True)

        self.kernel = torch.rand((self.num_filters, kernel_size, kernel_size),  requires_grad=True)

        # set padding variable
        self.padding = padding

        # set strides
        self.row_stride = stride[0]
        self.col_stride = stride[1]

        assert self.row_stride >= 1
        assert self.col_stride >= 1

        self.row_pad = 0
        self.col_pad = 0
        self.padding = padding
        if padding:
            self.row_pad = kernel_size // 2
            self.col_pad = kernel_size // 2



    def init_params(self, kernel=None, bias=None):
        """
        Initialize the layer parameters
        :return:
        """
        if bias is not None and len(bias.shape) != 1:
            print("invalid bias shape. not setting.")
            return
        if kernel is not None:
            biases_needed = kernel.shape[0]
        else:
            biases_needed = self.num_filters

        if bias is not None:
            biases_given = bias.shape[0]
        else:
            biases_given = self.bias.shape[0]

        if biases_given != biases_needed:
            print("Mismatch between number of filters and the number of biases given. Not setting."
                  "Bias should be 1-dimensional vector of size = no. of filters.")
            return

        if kernel is not None:
            self.kernel = kernel.clone().detach().requires_grad_()
        if bias is not None:
            self.bias = bias.clone().detach().requires_grad_()

        self.num_filters = self.kernel.shape[0]
        self.kernel_size = self.kernel.shape[1]

        if self.padding:
            self.row_pad = self.kernel_size // 2
            self.col_pad = self.kernel_size // 2


    def forward(self, input):
        """
        Forward pass
        :return:
        """
        row_stride = self.row_stride
        col_stride = self.col_stride
        kernel = self.kernel
        kernel_size = self.kernel_size
        num_filters = self.num_filters
        result_batch_size = input.shape[0]

        if self.padding:
            input = torch.nn.ZeroPad2d((self.row_pad, self.row_pad, self.col_pad, self.col_pad))(input)

        result_rows = 1 + (input.shape[-2] - kernel_size) // row_stride
        result_cols = 1 + (input.shape[-1] - kernel_size) // col_stride

        result = torch.zeros((result_batch_size, num_filters, result_rows, result_cols))

        # for k in range(result_batch_size):
        #     for c in range(num_filters):
        #         for i in range(0, result_rows):
        #             for j in range(0, result_cols):
        #                 result[k, c, i, j] = torch.sum(input[k,
        #                                                :,
        #                                                row_stride * i: row_stride * i + kernel_size,
        #                                                col_stride * j: col_stride * j + kernel_size]
        #                                                * kernel[c, :, :])
        # removed the out_channels / num_filters below.

        # for k in range(result_batch_size):
        for i in range(0, result_rows):
            for j in range(0, result_cols):
                result[:, :, i, j] = torch.sum(input[:,
                                               :,
                                               row_stride * i: row_stride * i + kernel_size,
                                               col_stride * j: col_stride * j + kernel_size]
                                               * kernel[:, :, :],
                                               (-1, -2)) + self.bias

        return result

    # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """