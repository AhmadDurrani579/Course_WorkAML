import torch
import torch.nn as nn

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

# You can add other lightweight transformation layers here, e.g., SEBlock, DepthwiseSeparableConv

class YOLOv8HeadWithTransform(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nl = len(anchors)
        self.na = anchors[0].shape[0] // 2 if anchors is not None and len(anchors) > 0 and isinstance(anchors[0], torch.Tensor) and anchors[0].numel() > 0 else (len(anchors[0]) // 2 if anchors is not None and len(anchors) > 0 and isinstance(anchors[0], list) and len(anchors[0]) > 0 else 0)
        self.no = nc + 5
        self.stride = torch.tensor([8., 16., 32.])
        self.export = False
        self.nc = nc
        self.reg_max = 16  # Dummy for loss compatibility

        self.convs = nn.ModuleList(nn.Conv2d(x, 256, 1) for x in ch[:-1])
        self.nl_convs = nn.ModuleList(nn.Conv2d(256 + ch[-1] if i > 0 else ch[-1], 256, 3, 1, 1) for i in range(self.nl))

        # Transformation Layer
        self.transform = PointwiseConv(256, 256) # Example: PointwiseConv - you can change this

        self.pred = nn.ModuleList(nn.Conv2d(256, self.no * self.na, 1) for _ in range(self.nl))

    def forward(self, x):
        out = []
        for i in range(self.nl):
            if i > 0:
                xi = self.convs[i - 1](x[-i - 1])
                x = self.nl_convs[i](torch.cat((x[-1], xi), 1))
            else:
                x = self.nl_convs[i](x[-1])

            # Apply Transformation
            x = self.transform(x)

            out.append(self.pred[i](x))

        return out if self.training else (torch.cat(out, 1),) if self.export else (torch.cat(out, 1),)