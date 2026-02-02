import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from mamba_ssm import Mamba
import torch_dct as dct
from pdb import set_trace as stx
import numbers
from functools import partial
from einops import rearrange

# from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

# from lkan.models import KANConv2d, KANLinear, KANLinearFFT

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
            nn.ReLU6(),
        )

class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super(ConvBN, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
        )

class Conv(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False
    ):
        super(Conv, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            )
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6(),
        )

class SeparableConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation="relu"):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights("normal")

    def init_weights(self, scheme=""):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=""):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == "normal":
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "trunc_normal":
            trunc_normal_tf_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            )
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == "relu":
        layer = nn.ReLU(inplace)
    elif act == "relu6":
        layer = nn.ReLU6(inplace)
    elif act == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == "gelu":
        layer = nn.GELU()
    elif act == "hswish":
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError("activation layer [%s] is not found" % act)
    return layer

class MambaLayer_new(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        # self.cab = CAB(dim)

    def forward(self, x):
        # print('x',x.shape)
        B, C, H, W = x.shape
        x_in = x
        x = x.reshape(B, H * W, C).contiguous()
        B, L, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        x_mamba = x_mamba.reshape(B, C, H, W).contiguous()
        # x_out = x_mamba + self.cab(x_in) * x_in
        return x_mamba

class MambaLayer(nn.Module):
    def __init__(
        self, in_chs=512, dim=128, d_state=16, d_conv=4, expand=2, last_feat_size=16
    ):
        super().__init__()
        pool_scales = self.generate_arithmetic_sequence(
            1, last_feat_size, last_feat_size // 8
        )
        self.pool_len = len(pool_scales)
        self.pool_layers = nn.ModuleList()
        self.pool_layers.append(
            nn.Sequential(
                ConvBNReLU(in_chs, dim, kernel_size=1), nn.AdaptiveAvgPool2d(1)
            )
        )
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvBNReLU(in_chs, dim, kernel_size=1),
                )
            )
        self.mamba = Mamba(
            d_model=dim * self.pool_len + in_chs,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    def forward(self, x):  # B, C, H, W
        res = x
        B, C, H, W = res.shape
        ppm_out = [res]
        for p in self.pool_layers:
            pool_out = p(x)
            pool_out = F.interpolate(
                pool_out, (H, W), mode="bilinear", align_corners=False
            )
            ppm_out.append(pool_out)
        x = torch.cat(ppm_out, dim=1)
        _, chs, _, _ = x.shape
        x = rearrange(x, "b c h w -> b (h w) c", b=B, c=chs, h=H, w=W)
        x = self.mamba(x)
        x = x.transpose(2, 1).view(B, chs, H, W)
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence

class ConvFFN(nn.Module):
    def __init__(self, in_ch=128, hidden_ch=512, out_ch=128, drop=0.0):
        super(ConvFFN, self).__init__()
        self.conv = ConvBNReLU(in_ch, in_ch, kernel_size=3)
        self.fc1 = Conv(in_ch, hidden_ch, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = Conv(hidden_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    def __init__(
        self,
        in_chs=512,
        dim=128,
        hidden_ch=512,
        out_ch=128,
        drop=0.1,
        d_state=16,
        d_conv=4,
        expand=2,
        last_feat_size=16,
    ):
        super(Block, self).__init__()
        self.mamba = MambaLayer(
            in_chs=in_chs,
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            last_feat_size=last_feat_size,
        )
        self.conv_ffn = ConvFFN(
            in_ch=dim * self.mamba.pool_len + in_chs,
            hidden_ch=hidden_ch,
            out_ch=out_ch,
            drop=drop,
        )

    def forward(self, x):
        x = self.mamba(x)
        x = self.conv_ffn(x)

        return x

class FeatureRefinementModule(nn.Module):
    def __init__(self, in_dim=128, down_kernel=5, down_stride=4):
        super().__init__()

        self.layer_norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        out_dim = in_dim
        self.lconv = nn.Conv2d(
            in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim
        )
        self.hconv = nn.Conv2d(
            in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim
        )
        self.norm1 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(in_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.down = nn.Conv2d(
            in_dim,
            in_dim,
            kernel_size=down_kernel,
            stride=down_stride,
            padding=down_kernel // 2,
            groups=in_dim,
        )
        self.proj = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x
        x = self.layer_norm1(x)
        dx = self.down(x)
        udx = F.interpolate(dx, size=(H, W), mode="bilinear", align_corners=False)
        lx = self.norm1(self.lconv(self.act(x * udx)))
        hx = self.norm2(self.hconv(self.act(x - udx)))

        out = self.act(self.proj(torch.cat([lx, hx], dim=1)))

        return out + x_in

class InceptionDWConv2d(nn.Module):
    """Inception depthweise convolution"""

    def __init__(
        self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125
    ):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = (in_channels - 4 * gc, gc, gc, gc, gc)
        self.mamba = MambaLayer_new(gc)

    def forward(self, x):
        # print("self.split_indexes are: ", self.split_indexes)
        x_id, x_hw, x_w, x_h, x_mamba = torch.split(x, self.split_indexes, dim=1)
        out = torch.cat(
            (
                x_id,
                self.dwconv_hw(x_hw),
                self.dwconv_w(x_w),
                self.dwconv_h(x_h),
                self.mamba(x_mamba),
            ),
            dim=1,
        )
        return out

class ConvolutionalGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MetaNeXtBlock(nn.Module):
    """MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        token_mixer=InceptionDWConv2d,
        norm_layer=nn.BatchNorm2d,
        mlp_layer=ConvolutionalGLU,
        mlp_ratio=4,
        act_layer=nn.GELU,
        ls_init_value=1e-6,
        drop_path=0.01,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        # self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.token_mixer(x)
        # x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

class LayerNorm(nn.Module):
    r"""From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class BottleNeck_Block(nn.Module):
    def __init__(self, channels, dims, multi_head=True, ffn=False):
        super(BottleNeck_Block, self).__init__()

        self.conv2 = nn.Conv2d(
            dims[1], channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv3 = nn.Conv2d(
            dims[2], channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv4 = nn.Conv2d(
            dims[3], channels, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv = nn.Conv2d(
            channels * 3, channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(channels * 3)

        self.qconv = MetaNeXtBlock(channels)
        print("dims are: ", dims)
        # self.conv2 = nn.Conv2d(
        #     channels, dims[1], kernel_size=3, stride=1, padding=1, bias=True
        # )
        # self.conv3 = nn.Conv2d(
        #     channels, dims[2], kernel_size=3, stride=1, padding=1, bias=True
        # )
        # self.conv4 = nn.Conv2d(
        #     channels, dims[3], kernel_size=3, stride=1, padding=1, bias=True
        # )

        self.gcn2 = FeatureRefinementModule(in_dim=channels)
        self.gcn3 = FeatureRefinementModule(in_dim=channels)
        self.gcn4 = FeatureRefinementModule(in_dim=channels)
        # self.dfconv = DeformableConv2d(channels,channels)

        # self.mlp1 = MLP(channels)
        # self.mlp2 = MLP(channels)
        # self.mlp3 = MLP(channels)
        # self.mlp4 = MLP(channels)

        self.act = nn.GELU()

    def forward(self, input2, input3, input4):
        B, C, H, W = input3.shape

        if input2.size()[2:] != input4.size()[2:]:
            input22 = F.interpolate(input2, size=input4.size()[2:], mode="bilinear")
        if input3.size()[2:] != input4.size()[2:]:
            input33 = F.interpolate(input3, size=input4.size()[2:], mode="bilinear")
        if input4.size()[2:] != input4.size()[2:]:
            input44 = F.interpolate(input4, size=input4.size()[2:], mode="bilinear")

        input22_a = self.gcn2(self.conv2(input22))
        input33_a = self.gcn3(self.conv3(input33))
        input44_a = self.gcn4(self.conv4(input4))
        fuse = torch.cat((input22_a, input33_a, input44_a), 1)
        fuse = self.act(self.conv(self.bn(fuse)))

        out = self.act(self.qconv(fuse))

        return out

class Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels=(64, 128, 256, 512),
        decoder_channels=128,
        num_classes=6,
        last_feat_size=16,
    ):
        super().__init__()
        print("encoder_channels is: ", encoder_channels)
        self.b3 = BottleNeck_Block(decoder_channels, encoder_channels)
        self.up_conv = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels * 2),
            nn.Upsample(scale_factor=2),
            InceptionDWConv2d(decoder_channels * 2),
            nn.Upsample(scale_factor=2),
            ConvBNReLU(decoder_channels * 2, decoder_channels),
            nn.Upsample(scale_factor=2),
        )
        self.pre_conv = ConvBNReLU(encoder_channels[0], decoder_channels)
        self.head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels // 2),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            # ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
            # nn.Upsample(scale_factor=2, mode="bilinear"),
            Conv(decoder_channels // 2, num_classes, kernel_size=1),
        )
        self.apply(self._init_weights)

    def forward(self, x0, x1, x2, x3):
        x3 = self.b3(x1, x2, x3)
        x3 = self.up_conv(x3)
        x = x3 + self.pre_conv(x0)
        x = self.head(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation="ReLU"):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class MyNet(nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        n_channels=3,
        pretrained=True,
        n_classes=6,
        decoder_channels=128,
        last_feat_size=32,
        img_size=224,
    ):
        super().__init__()
        self.inc = ConvBatchNorm(n_channels, 3)
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            output_stride=32,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
        )

        encoder_channels = self.backbone.feature_info.channels()
        print("encoder_channels is", encoder_channels)
        self.decoder = Decoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=n_classes,
            last_feat_size=last_feat_size,
        )

    def forward(self, x):
        # print("shape of input x is: ", x.shape)
        if x.size()[1] != 3:
            x = self.inc(x)
        # print("shape of x is: ", x.shape)
        x0, x1, x2, x3 = self.backbone(x)
        # x0 = x0.permute(0, 3, 1, 2)
        # x3 = x3.permute(0, 3, 1, 2)
        x = self.decoder(x0, x1, x2, x3)
        # print("shape of output x is: ", x.shape)
        return x

if __name__ == "__main__":
    model = MyNet(n_channels=3, n_classes=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch device: {device}")
    torch.cuda.empty_cache()
    model = model.to(device)
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Convert to millions
    number_in_millions = params / 1_000_000

    # Print the result
    print(f"{params} in millions are {number_in_millions} million.")

    from fvcore.nn import FlopCountAnalysis

    input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Example input tensor

    # Calculate FLOPs
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()

    # Convert FLOPs to GFLOPs
    gflops = total_flops / 1e9

    print(f"Total FLOPs: {total_flops}")
    print(f"GFLOPs: {gflops:.2f}")
