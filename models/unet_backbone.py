import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvBlock(nn.Module):
    """
    Symmetric Convolutional Block: Applies two successive 3x3 convolutions,
    each followed by BatchNorm2d and a localized ReLU activation layer.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetSatelliteSR(nn.Module):
    """
    Custom Symmetric U-Net Architecture configured for Satellite Super-Resolution.
    Transforms a [B, 1, 256, 256] input matrix into a [B, 1, 512, 512] target grid.
    """
    def __init__(self):
        super(UNetSatelliteSR, self).__init__()
        
        # 1. ENCODER PATH (Contracting Layers)
        self.inc = DoubleConvBlock(1, 64)       # Input layer maps 1 thermal channel
        self.down1 = nn.MaxPool2d(2)            # Resolves tensor down to 128x128
        self.conv1 = DoubleConvBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)            # Resolves tensor down to 64x64
        self.conv2 = DoubleConvBlock(128, 256)
        self.down3 = nn.MaxPool2d(2)            # Resolves tensor down to 32x32
        self.conv3 = DoubleConvBlock(256, 512)
        
        # 2. DECODER PATH (Expanding Reconstruction Layers)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = DoubleConvBlock(512, 256)  # 512 channels due to skip connection
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv5 = DoubleConvBlock(256, 128)  # 256 channels due to skip connection
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv6 = DoubleConvBlock(128, 64)   # 128 channels due to skip connection
        
        # 3. EXTRA TARGET RESOLUTION EXPANSION LAYER (200m -> 100m Mapping)
        # Doubles matrix grid layout coordinates from 256 to 512 target boundaries
        self.final_upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.outc = nn.Conv2d(64, 1, kernel_size=1) # Mapping back to single-channel TIR output

    def forward(self, x):
        # Encoder Operations with Feature Capture for Skip Connections
        x1 = self.inc(x)             # Save high-res map for lateral skip connection
        x2 = self.down1(x1)
        x2 = self.conv1(x2)          # Save mid-res map for lateral skip connection
        x3 = self.down2(x2)
        x3 = self.conv2(x3)          # Save low-res map for lateral skip connection
        x4 = self.down3(x3)
        x4 = self.conv3(x4)          # Bottleneck latency block
        
        # Decoder Operations with Tensor Concatenation (Skip Channels)
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1) # Lateral connection preserves edge dimensions
        x = self.conv4(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1) # Lateral connection preserves boundary lines
        x = self.conv5(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1) # Lateral connection preserves fine coastlines
        x = self.conv6(x)
        
        # Final Spatial Expansion Grid Transformation Block
        x = self.final_upsample(x)    # Converts spatial index array up to 512x512
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    # Operational matrix structure check for testing
    model = UNetSatelliteSR()
    dummy_input = torch.randn(1, 1, 256, 256) # Simulates 200m single-channel patch
    output = model(dummy_input)
    print(f"Input Matrix Dims:  {dummy_input.shape}")
    print(f"Output Matrix Dims: {output.shape} -> (Successfully Resolved to 100m Grid!)")
