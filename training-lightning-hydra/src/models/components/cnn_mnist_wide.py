from torch import nn


class ConvNetWide(nn.Module):
    """Wide convolutional network for 28x28 grayscale images (MNIST).

    Architecture:
        Conv2d(1, 64, 3, padding=1) -> BN -> ReLU -> MaxPool(2)   # 64x14x14
        Conv2d(64, 128, 3, padding=1) -> BN -> ReLU -> MaxPool(2)  # 128x7x7
        Conv2d(128, 256, 3, padding=1) -> BN -> ReLU               # 256x7x7
        Linear(256*7*7, 512) -> BN -> ReLU
        Linear(512, 10)
    """

    def __init__(self, lin1_size: int = 512, output_size: int = 10) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # Block 1: 1x28x28 -> 64x14x14
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 64x14x14 -> 128x7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 128x7x7 -> 256x7x7 (no pooling to preserve spatial detail)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Flatten + FC layers
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(inplace=True),
            nn.Linear(lin1_size, output_size),
        )

    def forward(self, x):
        return self.model(x)
