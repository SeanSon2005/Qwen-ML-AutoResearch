from torch import nn


class ConvNet(nn.Module):
    """Compact convolutional network for 28x28 grayscale images (MNIST).

    Architecture:
        Conv2d(1, 32, 3, padding=1) -> BN -> ReLU -> MaxPool(2)   # 32x14x14
        Conv2d(32, 64, 3, padding=1) -> BN -> ReLU -> MaxPool(2)  # 64x7x7
        Linear(64*7*7, 128) -> BN -> ReLU
        Linear(128, 10)
    """

    def __init__(self, lin1_size: int = 128, output_size: int = 10) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Flatten + FC layers
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(inplace=True),
            nn.Linear(lin1_size, output_size),
        )

    def forward(self, x):
        return self.model(x)
