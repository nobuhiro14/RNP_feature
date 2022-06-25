import torch
import torch.nn as nn
import torch.functional as F
import math


class base_model_cnn(nn.Module):
    def __init__(self):
        super(base_model_cnn, self).__init__()
        mx_channel = 512


        #VGG16 parameters
        #Note: one Fully connected layer was eliminated
        self.layers = nn.Sequential(
          nn.Conv2d(3,64,3,padding=1),
          nn.ReLU(True),
          nn.Conv2d(64,64,3,padding=1),
          nn.ReLU(True),
          nn.MaxPool2d(2,stride=2),

          nn.Conv2d(64,128,3,padding=1),
          nn.ReLU(True),
          nn.Conv2d(128,128,3,padding=1),
          nn.ReLU(True),
          nn.MaxPool2d(2,stride=2),

          nn.Conv2d(128,256,3,padding=1),
          nn.ReLU(True),
          nn.Conv2d(256,256,3,padding=1),
          nn.ReLU(True),
          nn.Conv2d(256,256,3,padding=1),
          nn.ReLU(True),
          nn.MaxPool2d(2,stride=2),

          nn.Conv2d(256,512,3,padding=1),
          nn.ReLU(True),
          nn.Conv2d(512,512,3,padding=1),
          nn.ReLU(True),
          nn.Conv2d(512,512,3,padding=1),
          nn.ReLU(True),
          nn.MaxPool2d(2,stride=2),

          nn.Conv2d(512,512,3,padding=1),
          nn.ReLU(True),
          nn.Conv2d(512,512,3,padding=1),
          nn.ReLU(True),
          nn.Conv2d(512,512,3,padding=1),
          nn.ReLU(True),
          nn.MaxPool2d(2,stride=2),
        )





        self.flatten = nn.Flatten()
        self.adaptive = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.layers(x)
        x = self.adaptive(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x
