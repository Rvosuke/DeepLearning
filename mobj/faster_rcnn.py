import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms


# %%
class MobjModel(nn.Module):
    def __init__(self):
        super(MobjModel, self).__init__()
        self.frcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    def forward(self, x):
        out = self.frcnn(x)
        return out


# %%
def img_preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).to(device='cuda')
    return img


# %%
def mobj_predict(img_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 图像预处理
    img_tensor = img_preprocess(img_path)
    img_tensor = img_tensor.to(device)

    # 加载模型
    model = MobjModel()
    model.eval()
    model.to(device)

    # 预测
    with torch.no_grad():
        predictions = model(img_tensor)

    # 获取预测结果
    boxes = predictions[0]['boxes'].data.cpu().numpy()
    labels = predictions[0]['labels'].data.cpu().numpy()
    scores = predictions[0]['scores'].data.cpu().numpy()

    # 打开图像一次
    img_PIL = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img_PIL)

    # 画框
    for i, box in enumerate(boxes):
        score = scores[i]
        if score < 0.8:
            continue

        class_index = str(labels[i].item())
        draw.rectangle(box, outline=getcolor(), width=3)
        draw.text((box[0], box[1]), text=class_index, fill='red')

    del draw

    # 保存图像
    img_PIL.save('mobj.jpg')

    return img_PIL


def getcolor():
    return np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)


# %%
if __name__ == '__main__':
    mobj_predict('csu.jpg')
