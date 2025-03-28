import torch.nn as nn
import torch
import torch.nn.functional as F
class MLP1d(nn.Module):
    def __init__(self, dim: int = 100, h_dim: int = 200) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(0.5),   
            nn.Tanh(),
            nn.Linear(h_dim, 4*h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = t.expand_as(x)
        x = torch.cat((x, t), dim=-1)
        return self.net(x)
class MLP2d(nn.Module):
    def __init__(self, dim: int = 100, h_dim: int = 1024) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(int(2 * dim**2), h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),   
            nn.Linear(h_dim, 4*h_dim),
            nn.ReLU(),
            nn.Linear(4*h_dim, int(dim**2))
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        n = x.shape[0]
        t = t.expand_as(x)
        x = torch.cat((x, t), dim=-1)

        x = x.view(n, -1)
        y = self.net(x)
        z = y.view(n, int(self.dim), int(self.dim)) 
        
        return z




class CNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, width=16):
        super().__init__()
        self.width = width
        self.device = "cuda"
        # Down-sampling layers
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm2d(width),
            nn.Sigmoid(),
            nn.Conv2d(width, width * 4, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm2d(width * 4),
            nn.Sigmoid(),
        )

        # Residual connection for down-sampling
        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels, width * 4, kernel_size=10, stride=4, padding=3),
            nn.BatchNorm2d(width * 4),
            nn.Sigmoid(),
        )

        # Placeholder for fully connected layers (will initialize dynamically)
        self.flat = nn.Identity()
        self.res2 = nn.Identity()

        # Up-sampling layers
        self.up = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm2d(width),
            nn.Sigmoid(),
            nn.ConvTranspose2d(width, out_channels, kernel_size=10, stride=2, padding=4),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # Expand time input and stack with spatial input
        t = t.expand_as(x).to(self.device)  # Ensure `t` is on the same device as `x`
        x = torch.stack([x, t], dim=1).to(self.device)  # Ensure `x` is on the same device as `t`

        # Down-sampling + residual connection
        y = self.down(x) + self.res1(x)

        # Flatten for fully connected layers
        n, c, h, w = y.shape
        y_flat = y.view(n, -1).to(self.device)  # Ensure `y_flat` is on the correct device

        # Dynamically initialize fully connected layers if not already
        if isinstance(self.flat, nn.Identity):
            flat_dim = c * h * w
            output_dim = x.shape[-2] * x.shape[-1]  # 输入 x 的分辨率
            
            self.flat = nn.Sequential(
                nn.Linear(flat_dim, flat_dim // 4),
                nn.Tanh(),
                nn.Linear(flat_dim // 4, flat_dim // 4),
                nn.Tanh(),
                nn.Linear(flat_dim // 4, flat_dim),
            ).to(self.device)  # Move `flat` layers to the correct device

            self.res2 = nn.Linear(flat_dim, flat_dim).to(self.device)  # Move `res2` to the correct device

        # Fully connected processing + residual connection
        y_flat = self.flat(y_flat) + self.res2(y_flat)

        # Reshape to convolutional dimensions
        y = y_flat.view(n, c, h, w)

        # Up-sampling to match input resolution
        z = self.up(y)
        z = z.squeeze(1) 
        return z
class FourierLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param modes: 傅立叶变换中保留的频率模式数量
        """
        super(FourierLayer2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # 实部和虚部的权重矩阵 (用于低频模式的线性操作)
        self.weights_real = nn.Parameter(torch.randn(in_channels, out_channels, modes, modes))
        self.weights_imag = nn.Parameter(torch.randn(in_channels, out_channels, modes, modes))

    def forward(self, x):
        # 对输入 x 进行傅立叶变换，形状为 (batch_size, channels, height, width)
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # 保留傅立叶变换的低频部分（前 self.modes 个频率模式）
        x_ft = x_ft[:, :, :self.modes, :self.modes]

        # 分别对实部和虚部应用权重矩阵
        x_ft_real = torch.einsum("bixy,ioxy->boxy", x_ft.real, self.weights_real) - torch.einsum("bixy,ioxy->boxy", x_ft.imag, self.weights_imag)
        x_ft_imag = torch.einsum("bixy,ioxy->boxy", x_ft.real, self.weights_imag) + torch.einsum("bixy,ioxy->boxy", x_ft.imag, self.weights_real)

        # 合并实部和虚部，得到更新后的傅立叶系数
        x_ft_updated = torch.complex(x_ft_real, x_ft_imag)

        # 执行逆傅立叶变换，将数据转换回空间域
        x = torch.fft.irfft2(x_ft_updated, s=(x.size(-2), x.size(-1)), norm='ortho')

        return x


class FNO(nn.Module):
    def __init__(self, in_channels = 2 , out_channels = 1, dim = 100, cutoff_ratio = 0.1, width = 64, layers = 4):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param dim: 输入数据的维度（例如，64表示64x64的输入）
        :param cutoff_ratio: 保留低频模式的比例
        :param width: 中间层的宽度
        :param layers: 傅立叶层的数量
        """
        super(FNO, self).__init__()
        self.width = width

        # 根据输入维度计算 modes
        modes = int(dim * cutoff_ratio)

        # 初始卷积层，将输入映射到一个高维空间
        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)

        # 创建多个傅立叶层
        self.fourier_layers = nn.ModuleList([FourierLayer2D(width, width, modes) for _ in range(layers)])

        # 最后的全连接层，将高维数据映射到输出空间
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):

#         """
        n = x.shape[0]
         # Expand t to match x's dimensions and concatenate along the channel axis
        t = t.expand_as(x)
        x = torch.stack([x, t], dim=1)  # Stack along channel dimension to create 2 input channels
        # 执行初始的卷积操作
        x = self.fc0(x)

        # 通过多个傅立叶层
        for layer in self.fourier_layers:
            x = layer(x)

        # 执行最后的卷积操作，将高维数据映射回到输出空间
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze(1)  # 移除通道维度，变为 (b, d, d)

        return x