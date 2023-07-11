import torch,collections
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 3, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512*BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



class VGG8(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG8, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG11(nn.Module):
    # it's VGG11
    def __init__(self, n_channels=3,n_class=10):
        super(VGG, self).__init__()
        self.features = []
        self.initial = None
        self.classifier = []
        self.layers = collections.OrderedDict()
        self.feature_dims=25088
        self.addFeatureBlock(name="11",n_channels=n_channels,out_channels=64,kernel_size=3,stride=1,padding=1,maxpoling=True)
        self.addFeatureBlock(name="12",n_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,maxpoling=True)
        self.addFeatureBlock(name="13",n_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,maxpoling=False)
        self.addFeatureBlock(name="14",n_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,maxpoling=True)
        self.addFeatureBlock(name="15",n_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,maxpoling=False)
        self.addFeatureBlock(name="16",n_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,maxpoling=True)
        self.addFeatureBlock(name="17",n_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,maxpoling=False)
        self.addFeatureBlock(name="18",n_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,maxpoling=True)
        self.avgpolling=nn.AdaptiveAvgPool2d(output_size=(7,7))
        self.addClassifierBlock("21",25088, 4096, activation=True, dropout=True)
        self.addClassifierBlock("22",4096, 4096, activation=True, dropout=True)
        self.addClassifierBlock("23",4096, n_class)
        self.initial_params = [param.data for param in self.parameters()]



    def addFeatureBlock(self,name,n_channels,out_channels,kernel_size,stride,padding,maxpoling):
        conv = nn.Conv2d(in_channels=n_channels,out_channels=out_channels, kernel_size=kernel_size,stride=stride,padding=padding)
        self.features.append(conv)
        self.layers['conv'+name] = conv
        setattr(self,'conv'+name,conv)

        ReLU = nn.ReLU(True)
        self.features.append(ReLU)
        self.layers['ReLU'+name] = ReLU
        setattr(self,'ReLU'+name,ReLU)

        if maxpoling:
            maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.append(maxpool)
            self.layers['pool'+name] = maxpool
            setattr(self,'pool'+name,maxpool)

    def addClassifierBlock(self,name,in_features,out_features,activation=False,dropout=False):
        linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.classifier.append(linear)
        self.layers['fc'+name] = linear
        setattr(self,'fc'+name,linear)

        if activation:
            ReLU = nn.ReLU(True)
            self.classifier.append(ReLU)
            self.layers['ReLU'+name] = ReLU
            setattr(self,'ReLU'+name,ReLU)

        if dropout:
            dropout = nn.Dropout(p=0.5)
            self.classifier.append(dropout)
            self.layers['dropout'+name] = dropout
            setattr(self,'dropout'+name,dropout)


    def forward(self, x, start=0, end=27):
        feautreslen=len(self.features)
        if start <= len(self.features)-1: # start in self.features
            for idx, layer in enumerate(self.features[start:]):
                # print(layer)
                x = layer(x)
                if idx == end:
                    return x
            x=self.avgpolling(x)
            x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                x = layer(x)
                if idx + feautreslen == end:
                    return x
        else:
            if start == feautreslen:
                x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                if idx >= start - feautreslen:
                    x = layer(x)
                if idx + feautreslen == end:
                    return x

    def get_params(self, end=17):
        params = []
        for layer in list(self.layers.values())[:end+1]:
            params += list(layer.parameters())
        return params

    def restore_initial_params(self):
        for param, initial in zip(self.parameters(), self.initial_params):
            param.data = initial

model=ResNet18()
print(model)
print(model(torch.rand((5,3,32,32))).shape)
