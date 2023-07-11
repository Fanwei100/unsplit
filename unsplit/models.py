import collections
import torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistNet(nn.Module):
    def __init__(self, n_channels=1):
        super(MnistNet, self).__init__()
        self.features = []
        self.initial = None
        self.classifier = []
        self.layers = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=8,
            kernel_size=5
        )
        self.features.append(self.conv1)
        self.layers['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(False)
        self.features.append(self.ReLU1)
        self.layers['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool1)
        self.layers['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5
        )
        self.features.append(self.conv2)
        self.layers['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(False)
        self.features.append(self.ReLU2)
        self.layers['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool2)
        self.layers['pool2'] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120)
        self.classifier.append(self.fc1)
        self.layers['fc1'] = self.fc1
     
        self.fc1act = nn.ReLU(False)
        self.classifier.append(self.fc1act)
        self.layers['fc1act'] = self.fc1act
     
        self.fc2 = nn.Linear(120, 84)
        self.classifier.append(self.fc2)
        self.layers['fc2'] = self.fc2
     
        self.fc2act = nn.ReLU(False)
        self.classifier.append(self.fc2act)
        self.layers['fc2act'] = self.fc2act
     
        self.fc3 = nn.Linear(84, 10)
        self.classifier.append(self.fc3)
        self.layers['fc3'] = self.fc3
        
        self.initial_params = [param.clone().detach().data for param in self.parameters()]

    def getName(self):
        return "MnistNet"


    def forward(self, x, start=0, end=10):
        if start <= 5: # start in self.features
            for idx, layer in enumerate(self.features[start:]):
                x = layer(x)
                if idx == end:
                    return x
            x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                x = layer(x)
                if idx + 6 == end:
                    return x
            return x
        else:
            if start == 6:
                x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                if idx >= start - 6:
                    x = layer(x)
                if idx + 6 == end:
                    return x
                
    def get_params(self, end=10):
        params = []
        for layer in list(self.layers.values())[:end+1]:
            params += list(layer.parameters())
        return params

    def restore_initial_params(self):
        for param, initial in zip(self.parameters(), self.initial_params):
            param.data = initial.requires_grad_(True)
            

class CifarNet(nn.Module):
    def __init__(self, n_channels=3):
        super(CifarNet, self).__init__()
        self.features = []
        self.initial = None
        self.classifier = []
        self.layers = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.features.append(self.conv11)
        self.layers['conv11'] = self.conv11

        self.ReLU11 = nn.ReLU(True)
        self.features.append(self.ReLU11)
        self.layers['ReLU11'] = self.ReLU11
        
        self.conv12 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.features.append(self.conv12)
        self.layers['conv12'] = self.conv12
        
        self.ReLU12 = nn.ReLU(True)
        self.features.append(self.ReLU12)
        self.layers['ReLU12'] = self.ReLU12

        self.pool1 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool1)
        self.layers['pool1'] = self.pool1

        self.conv21 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.features.append(self.conv21)
        self.layers['conv21'] = self.conv21

        self.ReLU21 = nn.ReLU(True)
        self.features.append(self.ReLU21)
        self.layers['ReLU21'] = self.ReLU21
        
        self.conv22 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.features.append(self.conv22)
        self.layers['conv22'] = self.conv22
        
        self.ReLU22 = nn.ReLU(True)
        self.features.append(self.ReLU22)
        self.layers['ReLU22'] = self.ReLU22

        self.pool2 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool2)
        self.layers['pool2'] = self.pool2
        
        self.conv31 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.features.append(self.conv31)
        self.layers['conv31'] = self.conv31
        
        self.ReLU31 = nn.ReLU(True)
        self.features.append(self.ReLU31)
        self.layers['ReLU31'] = self.ReLU31
        
        self.conv32 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.features.append(self.conv32)
        self.layers['conv32'] = self.conv32
        
        self.ReLU32 = nn.ReLU(True)
        self.features.append(self.ReLU32)
        self.layers['ReLU32'] = self.ReLU32
        
        self.pool3 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool3)
        self.layers['pool3'] = self.pool3
    
        self.feature_dims = 4 * 4 * 128
        self.fc1 = nn.Linear(self.feature_dims, 512)
        self.classifier.append(self.fc1)
        self.layers['fc1'] = self.fc1

        self.fc1act = nn.Sigmoid()
        self.classifier.append(self.fc1act)
        self.layers['fc1act'] = self.fc1act

        self.fc2 = nn.Linear(512, 10)
        self.classifier.append(self.fc2)
        self.layers['fc2'] = self.fc2

        self.initial_params = [param.data for param in self.parameters()]

    def getName(self):
        return "CifarNet"

    def forward(self, x, start=0, end=17):
        if start <= len(self.features)-1: # start in self.features
            for idx, layer in enumerate(self.features[start:]):
                x = layer(x)
                if idx == end:
                    return x
            x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                x = layer(x)
                if idx + 15 == end:
                    return x
            return x
        else:
            if start == 15:
                x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                if idx >= start - 15:
                    x = layer(x)
                if idx + 15 == end:
                    return x

    def get_params(self, end=17):
        params = []
        for layer in list(self.layers.values())[:end+1]:
            params += list(layer.parameters())
        return params

    def restore_initial_params(self):
        for param, initial in zip(self.parameters(), self.initial_params):
            param.data = initial

class VGG8(nn.Module):
    # it's VGG11
    def __init__(self, n_channels=3,n_class=10):
        super(VGG8, self).__init__()
        self.features = []
        self.classifier = []
        self.initial = None
        avaragpoolsize=(2,2)

        self.addFeatureBlock(name="11",n_channels=n_channels,out_channels=64,kernel_size=3,stride=1,padding=1,maxpoling=True)
        self.addFeatureBlock(name="12",n_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,maxpoling=True)
        self.addFeatureBlock(name="13",n_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,maxpoling=True)
        self.addFeatureBlock(name="14",n_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,maxpoling=True)
        self.addFeatureBlock(name="15",n_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1,maxpoling=True)
        self.feature_dims = self.features[-3].out_channels * avaragpoolsize[0] * avaragpoolsize[1]
        self.avgpolling=nn.AdaptiveAvgPool2d(output_size=avaragpoolsize)
        self.addClassifierBlock("21",self.feature_dims, 4096, activation=True, dropout=True)
        self.addClassifierBlock("22",4096, 4096, activation=True, dropout=True)
        self.addClassifierBlock("23",4096, n_class)
        self.initial_params = [param.data for param in self.parameters()]

    def getName(self):
        return "VGG8Net"

    def addFeatureBlock(self,name,n_channels,out_channels,kernel_size,stride,padding,maxpoling):
        conv = nn.Conv2d(in_channels=n_channels,out_channels=out_channels, kernel_size=kernel_size,stride=stride,padding=padding)
        self.features.append(conv)
        setattr(self,'conv'+name,conv)

        ReLU = nn.ReLU(True)
        self.features.append(ReLU)
        setattr(self,'ReLU'+name,ReLU)

        if maxpoling:
            maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.append(maxpool)
            setattr(self,'pool'+name,maxpool)

    def addClassifierBlock(self,name,in_features,out_features,activation=False,dropout=False):
        linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.classifier.append(linear)
        setattr(self,'fc'+name,linear)

        if activation:
            ReLU = nn.ReLU(True)
            self.classifier.append(ReLU)
            setattr(self,'ReLU'+name,ReLU)

        if dropout:
            dropout = nn.Dropout(p=0.5)
            self.classifier.append(dropout)
            setattr(self,'dropout'+name,dropout)


    def forward(self, x, start=0, end=-1):
        if end==-1: end=len(self.features)+len(self.classifier)-1
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
            return x
        else:
            if start == feautreslen:
                x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                if idx >= start - feautreslen:
                    x = layer(x)
                if idx + feautreslen == end:
                    return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.features = []
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.ReLU1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ReLU2 = nn.ReLU(True)
        self.features.extend([self.conv1,self.bn1,self.ReLU1,self.conv2,self.bn2,self.ReLU2])

        self.shortcut = []
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = [
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            ]

    def forward(self, x,start=0, end=-1):
        if end == -1: end = len(self.features)
        out=x
        processed=len(self.features)-3
        for i in range(min(end,5)):
            if start>i: continue
            out=self.features[i](out)
        if start==0 and end == len(self.features) and len(self.shortcut)>0:
            # resudal block will not work if split layer inside a block
            shout=x
            for i in range(min(end-processed,2)):
                shout = self.shortcut[i](shout)
                if processed+i>end: break
            out+=shout
        if end==len(self.features):
            out = self.features[-1](out)
        return out



class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1  = nn.ReLU(True)
        self.features=[(self.conv1,1),(self.bn1,1),(self.relu1,1)]
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool= nn.AvgPool2d(4)
        self.linear = nn.Linear(512*BasicBlock.expansion, num_classes)
        self.features.extend([(self.avgpool,1),(self.linear,1)])

    def getName(self):
        return "ResetNet"


    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
            self.features.append((layers[-1], len(layers[-1].features)))
        return nn.Sequential(*layers)

    def forward(self, x,start=0,end=-1):
        if end==-1: sum([l for f,l in self.features])
        out,processed=x,0
        for feature,nlayer in self.features[:-1]:
            processed+=nlayer
            if start>=processed:continue
            if nlayer==1:
                out=feature(out)
            else:
                startlocal,endlocal=0,-1
                if processed-nlayer<start and start<processed:
                    startlocal=start+nlayer-processed
                if processed>end:
                    endlocal=end+nlayer-processed
                    processed=end
                out = feature(out,start=startlocal,end=endlocal)
            if end==processed:
                return out

        out = out.view(out.size(0), -1)
        out = self.features[-1][0](out)

        return out


def PlotModel():
    print("MnistNet")
    print(MnistNet())
    print("CifarNet")
    print(CifarNet())
    print("VGG8Net")
    print(VGG8())


def getMiddleOutput():
    print(MnistNet())

#PlotModel()

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
# print(model)

# model=ResNet18()
# # out=model(torch.rand((2,3,32,32)))
# # out=model(torch.rand((2,3,32,32)),end=2)
# out=model(torch.rand((2,64,32,32)),start=2)
# print("Out",out.shape)

# model=BasicBlock(3,5)
# out=model(torch.rand((2,5,32,32)),start=5)
# print("Out",out.shape)



# #
# model=VGG8(3,10)
# print(model)
# out=model(torch.rand((2,3,128,128)))
# print("Out",out.shape)

# model=CifarNet(3)
# print(model)
# out=model(torch.rand((2,3,32,32)),end=0)
# print("Out",out.shape)
