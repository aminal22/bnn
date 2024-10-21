import torch
import torch.nn as nn

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        mean = torch.mean(input.abs(), dim=1, keepdim=True)
        input = input.sign()
        return input, mean

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Zero out gradients for the regions where input is not in the range [-1, 1]
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input, None  # Return None for the mean gradient since it's not needed

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        self.bn.weight.data.fill_(1.0)  # Initialize batch norm weights to 1
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x, _ = BinActive.apply(x)  # Use the static method of BinActive
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.xnor = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
            BinConv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
            BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)  # Prevent batch norm weights from being zero
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x
