import torch
import torch.nn.functional as F


class DebugEncoder(torch.nn.Module):
    """
    Debug encoder that encodes the input image to a 512 encoded latent space.
    """

    def forward_shape(self, x: torch.Tensor) -> int:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten()
        return x.shape[0]

    def __init__(self, image_size=256, latent_size=512):
        super(DebugEncoder, self).__init__()
        self.latent_size = latent_size

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.linear1 = torch.nn.Linear(self.forward_shape(torch.randn(1, 3, image_size, image_size)), latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.flatten()
        x = F.leaky_relu(self.linear1(x), 0.2)
        return x


def test_encoder():
    encoder = DebugEncoder()
    x = torch.randn(1, 3, 256, 256)
    print(encoder(x).shape)


if __name__ == '__main__':
    test_encoder()
