from torch import nn


class CatDogModel(nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.fcn1 = nn.Linear(256*12*12, 2048)
        self.fcn2 = nn.Linear(2048, 512)
        self.fcn3 = nn.Linear(512, 2)

    def forward(self, x):
        out = self.conv1(x)  # 224->224
        out = nn.ReLU(inplace=True)(out)
        out = nn.MaxPool2d(2, 2)(out)  # 224->112
        out = self.conv2(out)  # 112->108
        out = nn.ReLU(inplace=True)(out)
        out = nn.MaxPool2d(2, 2)(out)  # 108->54
        out = self.conv3(out)  # 54->52
        out = nn.ReLU(inplace=True)(out)
        out = nn.MaxPool2d(2, 2)(out)  # 52->26
        out = self.conv4(out)  # 26->24
        out = nn.ReLU(inplace=True)(out)
        out = nn.MaxPool2d(2, 2)(out)  # 24->12
        out = out.view(out.size(0), -1)  # (batch_size, 256, 12, 12)->(batch_size, 12*12*256)
        out = nn.Dropout(0.5)(out)
        out = self.fcn1(out)
        out = nn.ReLU(inplace=True)(out)
        out = nn.Dropout(0.5)(out)
        out = self.fcn2(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.fcn3(out)
        return out

    # 如此编写代码足够优雅，仅仅导入nn模块，不需要关注torch中的或者torch.nn.functional
    # 中的函数，只需要关注nn模块中的类，然后实例化，然后调用实例化对象的方法即可。
    # 如果借助F的函数式API, 会使得通篇的逻辑不同.
