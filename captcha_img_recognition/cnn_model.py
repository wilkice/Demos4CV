"""Define model
"""
# TODO: dropout or dropout2d

import torch.nn as nn
import setting


class Net(nn.Module):
    """CNN model to predict label
    3 CNN + 2 FC
    """
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear((setting.image_width // 8) *
                      (setting.image_height // 8) * 64, 1024),
            # nn.Dropout(0.5),
            nn.ReLU()
        )
        self.rfc = nn.Sequential(
            nn.Linear(1024, setting.char_num*setting.pool_length)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out
