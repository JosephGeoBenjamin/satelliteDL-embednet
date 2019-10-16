'''
Developed for tiles of Satellite Imagery
'''

import torch
from torch import nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view(self.shape)


class AutoEncoderVx(nn.Module):
    # Input Size 50x50 alone
    def __init__(self, img_channels=4):
        super(AutoEncoderVx, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 8, 3, stride=1, padding=1), # 50
            nn.Conv2d(8, 8, 3, stride=2, padding=1), #25
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),  #25
            nn.Conv2d(16, 16, 3, stride=2, padding=1), #13
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), #13
            nn.Conv2d(32, 32, 3, stride=2, padding=1), #7
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), #4
            Reshape(-1, 512, 1, 1),
        )
        self.decoder = nn.Sequential(
            Reshape(-1, 32, 4, 4),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),#7
            nn.ReLU(),
            nn.ConvTranspose2d(32,32, 3, stride=2, padding=1, output_padding=0), #13
            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1, output_padding=0), #13
            nn.ReLU(),
            nn.ConvTranspose2d(16,16, 3, stride=2, padding=1, output_padding=0), #25
            nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1, output_padding=0), #25
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1), #50
            nn.ConvTranspose2d(8, img_channels, 3, stride=1, padding=1, output_padding=0), #50
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


if __name__ =="__main__":

    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoderVx()
    model = model.to(device)
    summary(model, input_size=(3, 50, 50))
