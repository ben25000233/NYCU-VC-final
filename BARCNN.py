import torch
import torch.nn as nn

class BARCNN(nn.Module):
    def __init__(self, nb_filters, depth):
        super(BARCNN, self).__init__()
        self.nb_filters = nb_filters
        self.depth = depth

        # Initial layers
        self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(nb_filters)

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(nb_filters, nb_filters, kernel_size=3, stride=1, padding=1, bias=False) for _ in range(depth)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(nb_filters) for _ in range(depth)
        ])
        self.relu_layers = nn.ModuleList([
            nn.ReLU() for _ in range(depth)
        ])


        # Final layers
        self.conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, input_image):
        identity = input_image #input I
        x = self.conv1(input_image)
        x = self.bn1(x) 
        x = self.relu1(x)

        for i in range(self.depth ):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.relu_layers[i](x)

        x = self.relu2(self.conv2(x)) #output F(I)
        sec_x = x + identity #output H(I)
        
        self.temp = sec_x #output H(I)
        sec_x = self.conv1(sec_x)
        sec_x = self.bn1(sec_x) 
        sec_x = self.relu1(sec_x)
         
        for i in range(self.depth ):
            sec_x = self.conv_layers[i](sec_x)
            sec_x = self.bn_layers[i](sec_x)
            sec_x = self.relu_layers[i](sec_x)

        sec_x = self.relu2(self.conv2(sec_x)) #output g(F(I))
        output = sec_x + self.temp #output g(F(I)) + F(I) + I or g(F(I)) + H(I)
        
        return output





