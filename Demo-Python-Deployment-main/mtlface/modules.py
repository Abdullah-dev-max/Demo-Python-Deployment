import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

norm_layer = nn.InstanceNorm2d
act = lambda x: nn.LeakyReLU(0.2, True)



class SPPModule(nn.Module):
    def __init__(self, pool_mode='avg', sizes=(1, 2, 3, 6)):
        super().__init__()
        if pool_mode == 'avg':
            pool_layer = nn.AdaptiveAvgPool2d
        elif pool_mode == 'max':
            pool_layer = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError

        self.pool_blocks = nn.ModuleList([
            nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.pool_blocks]
        # print(f"{[x.shape for x in xs]}")
        x = torch.cat(xs, dim=1)
        # print(f"The X Output:{x.numel()},{x.dtype},{x.shape}")
        x = x.view(x.size(0), x.size(1), 1, 1)
        # print(f"The X Output:{x.numel()},{x.dtype},{x.shape}")
        return x


class AttentionModule(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super(AttentionModule, self).__init__()
        kernel_size = 7
        pool_size = (1, 2, 3)
        self.avg_spp = SPPModule('avg', pool_size)
        self.max_spp = SPPModule('max', pool_size)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

        _channels = channels * int(sum([x ** 2 for x in pool_size])) * 2
        self.channel = nn.Sequential(nn.Conv2d(_channels, _channels // reduction, kernel_size=1,padding=0,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(_channels // reduction, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )
        self.norm = nn.Identity()
        self.act = nn.Identity()

    def forward(self, x):
        # ke_si = (1, 2, 3)
        # print(f"Channels Calculation: {512 * int(sum([x ** 2 for x in ke_si])) * 2}")
        channel_input = torch.cat([self.avg_spp(x), self.max_spp(x)], dim=1)
        channel_scale = self.channel(channel_input)
        # print(f"Channel Layer:{self.channel}")
        # print(f"Channel Input: {channel_input.numel()},{channel_input.shape}")
        # print(f"Channel Layer Output:{channel_scale.numel()},{channel_scale.dtype},{channel_scale.shape}")
        
        spatial_input = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        spatial_scale = self.spatial(spatial_input)
        # print(f"Spatial Layers: {self.spatial}")
        # print(f"spatial Input: {spatial_input.numel()},{spatial_input.shape}")
        # print(f"Spatial Layer Output:{spatial_scale.numel()},{spatial_scale.dtype},{spatial_scale.shape}")
        scale = (channel_scale + spatial_scale) * 0.5
        # print(f"Scale:{scale.numel()},{scale.dtype},{scale.shape}")
        x_age = x * scale
        # print(f"X_ID:{x_id.numel()},{x_id.dtype},{x_id.shape}")
        x_id = self.act(self.norm(x_age))
        # print(f"X_ID:{x_id.numel()},{x_id.dtype},{x_id.shape}")
        x_id = x - x_age
        # print(f"X_Age:{x_age.numel()},{x_age.dtype},{x_age.shape}")
        return x_id, x_age


class Encoder(nn.Module):
    def __init__(self,age_group=7,repeat_num=4,input_size=112):
        super(Encoder, self).__init__()
        print("Encoder Module")
        self.n_styles = math.ceil(math.log(input_size, 2)) * 2 - 2
        facenet = IR_50(input_size=input_size)
        print(f"FaceNet:{facenet}")
        self.input_layer = facenet.input_layer
        self.block1 = facenet.body[0]
        self.block2 = facenet.body[1]
        self.block3 = facenet.body[2]
        self.block4 = facenet.body[3]
        self.output_layer = nn.Sequential(facenet.bn2, nn.Flatten(),facenet.dropout, facenet.fc, facenet.features)
        self.fsm = AttentionModule()


    def encode(self, x):
        # print(f"INput Layer:{self.input_layer}")
        # print(f"block1:{self.block1}")
        # print(f"block2:{self.block2}")
        # print(f"block3:{self.block3}")
        # print(f"block4:{self.block4}")
        # print(f"Oytput Layer:{ self.output_layer}")
        # print(f"Attention Module:{self.fsm}")
        x = self.input_layer(x)
        # print(f"Input_Layer Output:{x.numel()},{x.dtype},{x.shape}")
        c1 = self.block1(x)
        # print(f"Block1 Output:{c1.numel()},{c1.dtype},{c1.shape}")
        c2 = self.block2(c1)
        # print(f"Block2 Output:{c2.numel()},{c2.dtype},{c2.shape}")
        c3 = self.block3(c2)
        # print(f"Block3 Output:{c3.numel()},{c3.dtype},{c3.shape}")
        x = self.block4(c3)
        # print(f"Block4 Output:{x.numel()},{x.dtype},{x.shape}")
        # val = self.output_layer(x)
        # print(f"Output Layer: {val.numel()},{val.dtype},{val.shape}")
        x_id, x_age = self.fsm(x)
        # print(f"X_ID:{x_id.numel()},{x_id.dtype},{x_id.shape}")
        # print(f"X_Age:{x_age.numel()},{x_age.dtype},{x_age.shape}")
        x_vec = F.normalize(self.output_layer(x_age), dim=1)
        # print(f"X_Vec:{x_vec.numel()},{x_vec.dtype},{x_vec.shape}")
        return x_vec



class MTLFace(nn.Module):
    def __init__(self):
        super(MTLFace, self).__init__()
        input_size = 112
        self.encoder = Encoder(age_group=7,repeat_num=4,input_size=input_size)

    def encode(self, x):
        # print(self.encoder)
        x_vec1 = self.encoder.encode(x)
        # print(f"X_Vec_1 from encode : 1 {x_vec1.numel()},{x_vec1.shape}")
        x_vec2 = self.encoder.encode(torch.flip(x, dims=(3,)))
        # print(f"X_Vec_2 from encode : 2 {x_vec2.numel()},{x_vec2.shape}")
        x_vec = F.normalize(x_vec1 + x_vec2, dim=1)
        # print(f"Normalized Vector:{x_vec.numel()},{x_vec.shape}")
        return x_vec




class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


def get_block(unit_module, in_channel, depth, num_units, stride=2):
    layers = [unit_module(in_channel, depth, stride)] + [unit_module(depth, depth, 1) for _ in range(num_units - 1)]
    return nn.Sequential(*layers)


class IResNet(nn.Module):
    dropout_ratio = 0.4

    def __init__(self, input_size, num_layers, mode='ir', amp=False):
        super(IResNet, self).__init__()
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"

        if mode == 'ir':
            unit_module = bottleneck_IR
        else:
            raise NotImplementedError

        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))                         
 
        block1 = get_block(unit_module, in_channel=64, depth=64, num_units=num_layers[0])                                                                            
        print(unit_module)
        block2 = get_block(unit_module, in_channel=64, depth=128, num_units=num_layers[1])
        print(unit_module)
        block3 = get_block(unit_module, in_channel=128, depth=256, num_units=num_layers[2])
        print(unit_module)
        block4 = get_block(unit_module, in_channel=256, depth=512, num_units=num_layers[3])
        self.body = nn.Sequential(block1, block2, block3, block4)

        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)                         #(bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout(p=self.dropout_ratio, inplace=True)     #(dropout): Dropout(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * (input_size // 16) ** 2, 512)           #(fc): Linear(in_features=25088, out_features=512, bias=True)
        self.features = nn.BatchNorm1d(512, eps=1e-05)                    #(features): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self.amp = amp

        self._initialize_weights()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.amp):
            x = self.input_layer(x)
            x = self.body(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.amp else x)
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()


def IR_50(input_size, **kwargs):
    """Constructs a ir-50 model.
    """
    model = IResNet(input_size, [3, 4, 14, 3], 'ir', **kwargs)
    print(model)
    return model