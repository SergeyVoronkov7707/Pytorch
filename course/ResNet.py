from loader import nn, torch
from preparation import classes_train

class ResNet(nn.Module):
    def __init__(self, in_chnls, num_cls):
        super().__init__()

        self.conv1 = self.conv_block(in_chnls, 64, pool=True)  # 64x24x24
        self.conv2 = self.conv_block(64, 128, pool=True)  # 128x12x12
        self.resnet1 = nn.Sequential(self.conv_block(128, 128),
                                     self.conv_block(128, 128))  # Resnet layer 1: includes 2 conv2d

        self.conv3 = self.conv_block(128, 256, pool=True)  # 256x6x6
        self.conv4 = self.conv_block(256, 512, pool=True)  # 512x3x3
        self.resnet2 = nn.Sequential(self.conv_block(512, 512),
                                     self.conv_block(512, 512))  # Resnet layer 2: includes 2 conv2d

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Flatten(),
                                        nn.Linear(512, num_cls))  # num_cls

    @staticmethod
    def conv_block(in_chnl, out_chnl, pool=False, padding=1):
        layers = [
            nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_chnl),
            nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.resnet1(out) + out

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.resnet2(out) + out
        #         print(out.shape)
        #         print('*' * 20)
        return self.classifier(out)



net=ResNet(1, len(classes_train))
net.load_state_dict(torch.load('./models/gesture_detection_model_state_20_epochs.pth'))
net.eval()