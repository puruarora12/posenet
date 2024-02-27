import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
torch.set_printoptions(threshold=10_000)

def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock
        pre = f"inception_{key}/"
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            init(  pre+"1x1" ,nn.Conv2d(in_channels , n1x1 , kernel_size=1 , stride=1 , padding=0), weights),
            nn.ReLU()
        
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            init(  pre+"3x3_reduce" ,nn.Conv2d(in_channels , n3x3red , kernel_size=1 , stride=1 , padding=0), weights),
            nn.ReLU(),
            init(  pre+"3x3" ,nn.Conv2d(n3x3red , n3x3 , kernel_size=3 , stride=1 , padding=1), weights),
            nn.ReLU()
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            init(  pre+"5x5_reduce" ,nn.Conv2d(in_channels , n5x5red , kernel_size=1 , stride=1 , padding=0), weights),
            nn.ReLU(),
            init(  pre+"5x5" ,nn.Conv2d(n5x5red , n5x5 , kernel_size=5 , stride=1 , padding=2), weights),
            nn.ReLU()
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3 , stride=1 , padding=1),
            nn.ReLU(),
            init(pre+"pool_proj" , nn.Conv2d(in_channels  , pool_planes , kernel_size=1 , stride =1 , padding=0) , weights),
            nn.ReLU()
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
            x1 = self.b1(x)
            x2 = self.b2(x)
            x3 = self.b3(x)
            x4 = self.b4(x)
            x =torch.concat((x1, x2,x3,x4) ,dim =1)
            return x


class LossHeader(nn.Module):
    def __init__(self, in_channels, key, weights=None):
        super(LossHeader, self).__init__()

        # TODO: Define loss headers
        self.auxlayers = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            init(f'{key}/conv', nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
            nn.Flatten(),
            init(f'{key}/fc', nn.Linear(2048, 1024), weights),
            nn.Dropout(p=0.7)
        )

        self.fc1 = nn.Linear(1024, 3)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x):
        # TODO: Feed data through loss headers
        x = self.auxlayers(x)
        xyz = self.fc1(x)
        wpqr = self.fc2(x)
        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size = 5),
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0), weights),
            nn.ReLU(),
            init('conv2/3x3', nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), weights),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()

        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)
        self._3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64, "3b", weights)
        self._3mp = nn.MaxPool2d(kernel_size=3 , stride = 2 , padding=1)
        self._4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, "4a", weights)
        self._4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64, "4b", weights)
        self._4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64, "4c", weights)
        self._4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64, "4d", weights)
        self._4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128, "4e", weights)
        self._4mp = nn.MaxPool2d(kernel_size=3 , stride = 2 , padding=1)
        self._5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128, "5a", weights)
        self._5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128, "5b", weights)
        self.postlayer =nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024,2048),
            nn.Dropout(p=0.4)
        )

        self.fc1 = nn.Linear(2048 , 3)
        self.fc2 = nn.Linear(2048 , 4)
        self.lh1 = LossHeader(512 , "loss1" , weights)
        self.lh2 = LossHeader(528 , 'loss2' , weights)
        self.ReLU = nn.ReLU()


        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        x = self.pre_layers(x)
        x3a = self._3a(x)
        x3b = self._3b(x3a)
        x3mp =self._3mp(x3b)
        x3mp = self.ReLU(x3mp)
        x4a = self._4a(x3mp)
        x4b = self._4b(x4a)
        loss1_xyz , loss1_wpqr = self.lh1(x4a)
        x4c = self._4c(x4b)
        x4d = self._4d(x4c)
        x4e = self._4e(x4d)
        loss2_xyz, loss2_wpqr =  self.lh2(x4d)
        x4mp = self._4mp(x4e)
        x4mp = self.ReLU(x4mp)
        x5a = self._5a(x4mp)
        x5b = self._5b(x5a)
        out = self.postlayer(x5b)
        loss3_xyz = self.fc1(out)
        loss3_wpqr = self.fc2(out)





        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr


class PoseLoss(nn.Module):

    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr


    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        loss =0
        
        gt_xyz = poseGT[: , :3]
        gt_wpqr = poseGT[:, 3:]
        
            
        gt_normalized = F.normalize(gt_wpqr , p=2.0, dim =1)
        loss1_xyz = F.mse_loss(p1_xyz,gt_xyz)
        # print(f"p1_wpqr shape {p1_wpqr[i].shape} gt shape {gt_normalized.shape}")
        
        loss1_wpqr = self.w1_wpqr*(F.mse_loss(p1_wpqr,gt_wpqr/gt_normalized ))
        loss1 = (loss1_xyz + loss1_wpqr)*self.w1_xyz

        loss2_xyz = F.mse_loss(p2_xyz,gt_xyz)
        loss2_wpqr = self.w2_wpqr*(F.mse_loss(p2_wpqr, gt_wpqr/gt_normalized))
        loss2 = (loss2_xyz + loss2_wpqr)*self.w2_xyz

        loss3_xyz = F.mse_loss(p3_xyz,gt_xyz)
        loss3_wpqr = self.w3_wpqr*(F.mse_loss(p3_wpqr,  gt_wpqr/gt_normalized))
        loss3 = (loss3_xyz + loss3_wpqr)*self.w3_xyz
        loss = loss1+loss2+loss3
        # loss= loss/poseGT.size()[0]
        
        return loss
