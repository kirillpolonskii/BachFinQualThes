import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # input: N x channels_img+1 x 256 x 256
            nn.Conv2d(channels_img + 1, features_d, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            # N x channels_img x 128 x 128
            self._block(features_d, features_d * 2, 3, 2, 1),
            nn.Dropout2d(0.3),
            self._block(features_d * 2, features_d * 4, 3, 2, 1),
            nn.Dropout2d(0.3),
            self._block(features_d * 4, features_d * 8, 3, 2, 1),
            nn.Dropout2d(0.3),
            self._block(features_d * 8, features_d * 16, 3, 2, 1),
            nn.Dropout2d(0.3),
            # After all _block img output is 8x8 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 16, 1, kernel_size=8, stride=2, padding=0),
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)  # N x C x img_size x img_size
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            # Input: N x channels_noise + embed_size x 1 x 1
            self._block(channels_noise + embed_size, features_g * 32, 8, 1, 0, 0),
            nn.Dropout2d(0.3),
            self._block(features_g * 32, features_g * 16, 3, 2, 1, 1),
            nn.Dropout2d(0.3),
            self._block(features_g * 16, features_g * 8, 3, 2, 1, 1),
            nn.Dropout2d(0.3),
            self._block(features_g * 8, features_g * 4, 3, 2, 1, 1),
            nn.Dropout2d(0.3),
            self._block(features_g * 4, features_g * 2, 3, 2, 1, 1),
            nn.Dropout2d(0.3),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=3, stride=2, padding=1, output_padding=1
            ), # Output: N x channels_img x 256x256
            nn.Sigmoid(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 1, 256, 256
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    labels = torch.IntTensor([1, 2, 3, 4, 5, 6, 7, 8])
    print(labels)
    disc = Discriminator(in_channels, 8, 30, H)
    print(disc(x, labels).shape)
    assert disc(x, labels).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8, 30, H, 100)
    z = torch.randn((N, noise_dim, 1, 1))
    print('gen(z, labels).shape ', gen(z, labels).shape)
    assert gen(z, labels).shape == (N, in_channels, H, W), "Generator test failed"


# test()

