import glob
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class CatDogDataset(Dataset):
    def __init__(self, train=True):
        file_dir = 'cat_dog_data/training_set' if train else 'cat_dog_data/test_set'
        self.files = []
        self.labels = []

        for category in ['cats', 'dogs']:
            path = os.path.join(file_dir, category, '*.jpg')
            files = glob.glob(path)
            self.files.extend(files)
            label = 1 if category == 'dogs' else 0
            self.labels.extend([label] * len(files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        label = self.labels[index]

        image = Image.open(file).convert('RGB')
        image = transform(image)
        return image, label
