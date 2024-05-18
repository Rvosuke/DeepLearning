import os
import cv2
import time
import torch
import numpy as np
import albumentations as A
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.segmentation import fcn_resnet50
from matplotlib import pyplot as plt
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, classification_report


class BreastCancerSegmentationDataset(Dataset):
    """
    乳腺癌分割数据集
    """
    def __init__(self, img_dir, mask_dir, transform=None, one_hot_encode=True, target_size=(512, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.one_hot_encode = one_hot_encode
        self.target_size = target_size
        self.img_filenames = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        # 根据文件存放方式设置os，便于建立原始图像与mask图像的联系
        img_name = self.img_filenames[index]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name[:-4] + '_mask'+img_name[-4:])
        # 跳过 .ipynb_checkpoints 文件
        if img_path.endswith(".ipynb_checkpoints") or mask_path.endswith(".ipynb_checkpoints"):
            return self.__getitem__((index + 1) % len(self))
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 适应度处理，检查是否成功将图像添加入os
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")

        # 将 image 和 mask 的图像进行强制转化，转化成同样大小
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # 将mask进行独热处理
        if self.one_hot_encode:
            mask = self.one_hot_encoding(mask, num_classes=3)

        # 定义transform构架，为数据增强做准备
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 将mask的属性值转化为float类型
        mask = np.asarray(mask)   # 转换为NumPy数组
        mask = mask.astype(np.float32)   # 变换dtype
        mask = torch.from_numpy(mask) # 转换为Tensor

        return image, mask

    def one_hot_encoding(self, mask, num_classes):
        one_hot = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype=np.uint8)
        for c in range(num_classes):
            one_hot[..., c] = (mask == c)
        return one_hot


class BreastCancerClassificationDataset(Dataset):
    """乳腺癌分类数据集"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_filenames = os.listdir(img_dir)
        self.target_size = (512, 512)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        img_name = self.img_filenames[index]
        img_path = os.path.join(self.img_dir, img_name)
        if img_path.endswith(".ipynb_checkpoints"):
            return self.__getitem__((index + 1) % len(self))
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")


        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        if 'benign' in img_name:
            label = 0
        elif 'malignant' in img_name:
            label = 1
        elif 'normal' in img_name:
            label = 2

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# 训练集的transmform构架，为了增强泛化能力，进行了水平翻转，随机亮度对比度调整，随机旋转，随机裁剪和缩放，归一化，将图像数据转换为PyTorch张量等操作
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.3),
    A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), p=0.2),
    A.Normalize(),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

# 构建数据集字典，便于数据加载
SegmentationDataset = {
    'train': BreastCancerSegmentationDataset("image", "mask", transform=train_transform),
    'test': BreastCancerSegmentationDataset("image", "mask", transform=test_transform)
}
# 构建训练集和测试集的 DataLoader
S_train_loader = DataLoader(SegmentationDataset['train'], batch_size=32, shuffle=True, num_workers=8, drop_last=True)
S_test_loader = DataLoader(SegmentationDataset['test'], batch_size=32, shuffle=False, num_workers=8, drop_last=True)

ClassificationDataset = {
    'train': BreastCancerClassificationDataset("image", transform=train_transform),
    'test': BreastCancerClassificationDataset("image", transform=test_transform)
}

C_train_loader = DataLoader(ClassificationDataset['train'], batch_size=32, shuffle=True, num_workers=0, drop_last=True)
C_test_loader = DataLoader(ClassificationDataset['test'], batch_size=32, shuffle=False, num_workers=0, drop_last=True)


class UNetTrainer:
    """实际上我们使用的是一个全卷积网络（FCN）的ResNet50实现，而不是U-Net"""
    def __init__(self, num_classes=3, lr=1e-4):
        # 我们使用的fcn_resnet50是U-Net模型的变体,其中编码器部分初始化了ResNet50的权重。这可以加速模型训练和提高最终性能。但解码器部分仍需要我们从零训练
        self.model = fcn_resnet50(pretrained=False, num_classes=num_classes)
        # 损失函数使用交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss()
        # 使用adam优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # 设置GPU为训练设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.5)  # 设置学习率衰减的调度器，每过5次迭代学习率衰减0.5

    def calculate_iou(self, pred, target):
        """IoU计算函数"""
        return jaccard_score(target.cpu().numpy().ravel(), pred.cpu().numpy().ravel(), average="macro")

    def evaluate(self, dataloader, phase):
        """评估函数"""
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        # 初始化结果参数
        running_loss = 0.0
        running_iou = 0.0
        running_f1_score = 0.0

        for images, masks in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # 这里将mask中每个像素的最大值所在的通道作为像素的类别，并将mask转换为long类型。这样处理可以保持输出preds与masks的维度匹配，以便后续计算损失、IoU和F1分数。
            # masks = torch.mean(masks, dim=3, keepdim=False).long() 取均值会引起精度异常高，并不是真的高，是过于乐观了，骗骗小孩子的那种(*^_^*)
            masks = torch.argmax(masks, dim=3, keepdim=False).long()

            # 梯度清零
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                # 得出模型预测结果
                outputs = self.model(images)['out']
                preds = torch.argmax(outputs, dim=1)  # issue：你他妈不觉得这里和上边的masks对不上号？
                # 计算损失值
                loss = self.criterion(outputs, masks)

                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_iou += self.calculate_iou(preds, masks.data)
            running_f1_score += f1_score(masks.cpu().numpy().ravel(), preds.cpu().numpy().ravel(), average="macro")

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_iou = running_iou / len(dataloader)
        epoch_f1_score = running_f1_score / len(dataloader)

        print(f"{phase} Loss: {epoch_loss:.4f} IoU: {epoch_iou:.4f} F1: {epoch_f1_score:.4f}")


    def train(self, num_epochs, train_loader, test_loader=None):
        """模型训练函数"""

        for epoch in range(num_epochs):
            print("-" * 20)
            print(f"Epoch {epoch + 1}/{num_epochs}")

            start_time = time.time()

            self.evaluate(train_loader, "train")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Epoch time: {elapsed_time:.4f}s")
            self.scheduler.step()  # 学习率衰减

            if (epoch + 1) % 5==0 and test_loader:
                self.evaluate(test_loader, "test")

        print("Training complete")


        torch.save(self.model.state_dict(), 'S_trained_model.pth')


    def test(self, test_loader, model_path='S_trained_model.pth'):
        """模型测试函数"""
        # 加载模型
        self.model.load_state_dict(torch.load(model_path))

        # 评估模型在测试集上的性能，即估计模型的泛化性能
        print("Evaluating the model on the test dataset")
        self.evaluate(test_loader, "test")


class ResNetTrainer:
    def __init__(self, learning_rate=0.01):
        self.num_classes = 3
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

        self.patience = 3  # 设定连续多少个epochs未出现验证损失改善时停止训练
        self.best_val_loss = float("inf")  # 初始化最佳验证损失为无穷大

    def early_stopping(self, current_val_loss):
        """早停法"""
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            return True

        return False

    def train(self, epochs, train_loader):
        self.model.train()
        train_losses = []
        start_time = time.time()
        for epoch in range(epochs):
            print('-' * 50)
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            train_losses.append(running_loss / len(train_loader))
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
            if self.early_stopping(running_loss):  # 使用验证损失调用早停法
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            self.scheduler.step()
        total_time = time.time() - start_time
        print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        torch.save(self.model.state_dict(), 'C_trained_model.pth')
        self.plot_training_loss(train_losses, epochs)

    def test(self, test_loader):
        self.model.eval()
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())

        accuracy = accuracy_score(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels)

        print(f"Accuracy: {accuracy}")
        print(report)

    def plot_training_loss(self, training_losses, num_epochs):
        """绘制训练loss和精度曲线"""
        epochs = list(range(1, num_epochs + 1))
        plt.plot(epochs, training_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


Strainer = UNetTrainer()
Strainer.train(20, S_train_loader, S_test_loader)
Strainer.test(S_test_loader)

Ctrainer = ResNetTrainer(0.001)
Ctrainer.train(20, C_train_loader)
Ctrainer.test(C_test_loader)