import math

import torch.nn as nn
import torch
import torchvision.models.resnet as rn
from torchvision.utils import _log_api_usage_once
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
import torchvision

class PConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 p: float = 1.0,
                 keep_dim_in:bool = False,
                 keep_dim_out:bool = False,
                 ):
        super(PConv2d, self).__init__(
            in_channels if keep_dim_in else int(in_channels * p),
            out_channels if keep_dim_out else int(out_channels * p),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

class PLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=None, p: float = 1.0, keep_dim_in:bool = False, keep_dim_out:bool = False,):
        super(PLinear, self).__init__(in_features if keep_dim_in else int(in_features * p) ,
                                      out_features if keep_dim_out else int(out_features * p), bias=bias, device=device, dtype=dtype)

class PBasicBlock(rn.BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
        p: float = 1.0,
        keep_dim_in: bool = False,
        keep_dim_out: bool = False,
    ):
        super().__init__(
            inplanes=inplanes if keep_dim_in else int(inplanes * p),
            planes=planes if keep_dim_out else int(planes * p),
            stride=stride,
            downsample = downsample if (downsample is not None or (not keep_dim_out)) else nn.Conv2d(int(inplanes * p), planes, 1, bias=False),
            groups= groups,
            base_width= base_width,
            dilation= dilation,
            norm_layer = norm_layer,
        )

class PBottleneck(rn.Bottleneck):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
        p: float = 1.0,
        keep_dim_in: bool = False,
        keep_dim_out: bool = False,
    ) -> None:
        super().__init__(
            inplanes=inplanes if keep_dim_in else int(inplanes * p),
            planes=planes if keep_dim_out else int(planes * p),
            stride=stride,
            downsample = downsample if (not keep_dim_out and downsample is None) else nn.Conv2d(int(inplanes * p), planes, 1, bias=False),
            groups= groups,
            base_width= base_width,
            dilation= dilation,
            norm_layer = norm_layer,
        )

class PResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
        p:float = 1.0,
    ):
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.p = p
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = PConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, keep_dim_in=True, p=p)
        self.bn1 = norm_layer(int(self.inplanes*p))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = PLinear(512 * block.expansion, num_classes, p=p, keep_dim_out=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, PBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                rn.conv1x1(int(self.inplanes*self.p), int(planes * block.expansion*self.p), stride),
                norm_layer(int(planes * block.expansion*self.p)),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, p=self.p
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    p=self.p
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class PResNetEncoder(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
        p:float = 1.0,
    ):
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.p = p
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = PConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, keep_dim_in=True, p=p)
        self.bn1 = norm_layer(int(self.inplanes*p))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PBottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, PBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                rn.conv1x1(int(self.inplanes*self.p), int(planes * block.expansion*self.p), stride),
                norm_layer(int(planes * block.expansion*self.p)),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, p=self.p
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    p=self.p
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        return self._forward_impl(x)

class PTransformerBlock(nn.Module):
    def __init__(self, num_attn_heads, input_dim, attn_hidden_dim, fc_hidden_dim, dropout=0., p=1.0):
        super(PTransformerBlock, self).__init__()
        self.input_dim = input_dim
        self.k = int(attn_hidden_dim*p)
        self.num_heads = num_attn_heads
        self.p = p

        self.wq = PLinear(input_dim, num_attn_heads * attn_hidden_dim, bias=False, p=1.0)
        self.wk = PLinear(input_dim, num_attn_heads * attn_hidden_dim, bias=False, p=1.0)
        self.wv = PLinear(input_dim, num_attn_heads * attn_hidden_dim, bias=False, p=1.0)
        self.wc = PLinear(num_attn_heads * attn_hidden_dim, input_dim, bias=False, p=1.0)
        self.dropout_attn = nn.Dropout(dropout)

        self.w1 = PLinear(input_dim, fc_hidden_dim, p=p)
        self.dropoutfc = nn.Dropout(dropout)
        self.w2 = PLinear(fc_hidden_dim, input_dim, p=p)

        self.layernorm1 = nn.LayerNorm(int(p*input_dim))
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(int(p*input_dim))
        self.dropout2 = nn.Dropout(dropout)

        nn.init.normal_(self.wq.weight, 0, .02)
        nn.init.normal_(self.wk.weight, 0, .02)
        nn.init.normal_(self.wv.weight, 0, .02)
        nn.init.normal_(self.wc.weight, 0, .02)

        nn.init.normal_(self.w1.weight, 0, .02)
        nn.init.constant_(self.w1.bias, 0.0)
        nn.init.normal_(self.w2.weight, 0, .02)
        nn.init.constant_(self.w2.bias, 0.0)

    def forward(self, x, mask):
        # x: (seq_len, B, d); mask: (seq_len, seq_len)
        # mask is 0 or -inf. Add to pre-softmax scores
        seq_len, batch_size, embed_dim = x.shape
        query = self.wq(x).contiguous().view(seq_len, batch_size * self.num_heads, self.k).transpose(0, 1)  # (seq_len, B*H, k)
        key  = self.wk(x).contiguous().view(seq_len, batch_size * self.num_heads, self.k).transpose(0, 1)  # (seq_len, B*H, k)
        value = self.wv(x).contiguous().view(seq_len, batch_size * self.num_heads, self.k).transpose(0, 1) # (seq_len, B*H, k)
        # Apply attention
        alpha = torch.bmm(query, key.transpose(1, 2)) + mask  # (seq_len, B*H, B*H)
        alpha = F.softmax(alpha / math.sqrt(self.k), dim=-1)  # (seq_len, B*H, B*H)
        alpha = self.dropout_attn(alpha)  # (seq_len, B*H, B*H)
        u = torch.bmm(alpha, value)  # (seq_len, B*H, k)
        u = u.transpose(0, 1).contiguous().view(seq_len, batch_size, self.num_heads*self.k)   # (seq_len, B, H*k)
        # Apply first FC (post-attention)
        u = self.dropout1(self.wc(u))  # (seq_len, B, d)
        # Apply skip connection
        u = x + u  # (seq_len, B, d)
        # Apply layer norm
        u = self.layernorm1(u)  # (seq_len, B, d)
        # Apply FC x 2
        z = self.dropout2(self.w2(self.dropoutfc(F.relu(self.w1(u)))))  # (seq_len, B, d)
        # Apply skip connection
        z = u + z  # (seq_len, B, d)
        # Apply layer norm
        z = self.layernorm2(z)
        return z  # (seq_len, B, d)

    def add_dropout(self, dropout):
        self.dropout_attn = nn.Dropout(dropout)
        self.dropoutfc = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

class PWordLMTransformer(nn.Module):
    def __init__(self, seq_len=20, vocab_size=10004, input_dim=256, attn_hidden_dim=64, fc_hidden_dim=1024,
                 num_attn_heads=4, num_layers=4, dropout_tr=0., dropout_io=0.,p=1.0
    ):
        super().__init__()
        self.p = p
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mask = None
        self.pos = None
        self.dims = input_dim
        self.dropout = dropout_tr
        self.positional_embedding = nn.Embedding(seq_len, input_dim)
        self.drop_i = nn.Dropout(dropout_io)
        self.word_embedding = nn.Embedding(vocab_size, input_dim)
        self.fc1 = PLinear(input_dim, input_dim, keep_dim_in=True, p=p)
        self.transformer = nn.ModuleList()
        for i in range(num_layers):
            self.transformer.append(PTransformerBlock(num_attn_heads, input_dim, attn_hidden_dim, fc_hidden_dim, dropout_tr, p=p))

        self.head = PLinear(input_dim, vocab_size, bias=False, keep_dim_out=True, p=p)  # bias vector below
        self.drop_o = nn.Dropout(dropout_io)
        self.bias = nn.Parameter(torch.ones(vocab_size))

        nn.init.normal_(self.positional_embedding.weight, 0, .02)
        nn.init.normal_(self.word_embedding.weight, 0, .02)
        if not self.tied_weights: nn.init.normal_(self.head.weight, 0, .02)
        nn.init.constant_(self.bias, 0.0)
        self.is_on_client = None
        self.is_on_server = None

    def forward(self, x):
        # x: (seq_len, batch_size)
        if self.mask is None or self.mask.shape[0] != x.shape[0]:
            self.mask = torch.triu(torch.ones(len(x), len(x)))
            self.mask.masked_fill_(self.mask == 0, float('-inf')).masked_fill_(self.mask == 1, float(0.0))
            self.mask = self.mask.transpose(0, 1).to(x.device)
            self.pos = torch.arange(0, x.shape[0], dtype=torch.long).to(x.device)

        x = self.word_embedding(x) * math.sqrt(self.dims)
        p = self.positional_embedding(self.pos)[:, None, :]
        z = F.relu(self.drop_i(x) + self.drop_i(p))
        z = self.fc1(z)
        for layer in self.transformer:
            z = layer(z, self.mask)

        z = self.drop_o(z)
        outputs = self.head(z)
        return outputs + self.bias  # pre-softmax weights

class PCNNEncoder(nn.Module):
    def __init__(self, p: float = 1.0):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool2d = nn.MaxPool2d(2)
        self.conv1 = PConv2d(3, 64, 5, bias=True, keep_dim_in=True, p=p)
        self.conv2 = PConv2d(64, 64, 5, bias=True, p=p)
        self.flatten = nn.Flatten(1)
        self.linear1 = PLinear(1600, 384, bias=True, p=p)
        self.linear2 = PLinear(384, 192, bias=True, p=p)

    def forward(self, x):
        x = self.pool2d(self.relu(self.conv1(x)))
        x = self.pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x

class PAlexNetEncoder(nn.Module):
    def __init__(self, p: float = 1.0):
        super().__init__()
        self.p = p
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = PConv2d(3, 64, kernel_size=11, stride=4, padding=2, p=p, keep_dim_in=True)
        self.conv2 = PConv2d(64, 192, kernel_size=5, padding=2, p=p)
        self.conv3 = PConv2d(192, 384, kernel_size=3, padding=1, p=p)
        self.conv4 = PConv2d(384, 256, kernel_size=3, padding=1, p=p)
        self.conv5 = PConv2d(256, 256, kernel_size=3, padding=1, p=p)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = PLinear(256 * 6 * 6, 1024, p=p)
        self.fc2 = PLinear(1024, 1024, p=p)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(self.relu(self.conv5(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

# if __name__ == '__main__':
#     conv = PConv2d(3, 64, kernel_size=5, keep_dim_in=True, p=0.5)
#     x = torch.randn(3, 3, 32, 32)
#     y = conv(x)
#     print(y.shape)
#     x1 =torch.randn(2, 10)
#     fc = PLinear(10, 64, keep_dim_in=True, p=0.5)
#     y1 = fc(x1)
#     print(y1.shape)
#
#     resnet25 = PResNet(PBasicBlock, [2,2,2,2], norm_layer=lambda x: nn.GroupNorm(2, x), p=0.25, num_classes=10)
#     resnet50 = PResNet(PBasicBlock, [2,2,2,2], norm_layer=lambda x: nn.GroupNorm(2, x), p=0.5, num_classes=10)
#     y1 = resnet25(x)
#     y2 = resnet50(x)
#     print(y1.shape)
#     print(y2.shape)