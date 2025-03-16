from torch.utils.data import Dataset
from torchvision import transforms

class MNISTDataset(Dataset):
    def __init__(self, dataset, augment=True):
        """
        :param dataset: 原始MNIST数据集对象
        :param augment: 是否启用数据增强
        """
        self.images = dataset.data.float() / 255.0  # 归一化到[0,1]
        self.labels = dataset.targets
        self.transform = transforms.Compose([
            transforms.RandomRotation(15),      # 随机旋转
        ]) if augment else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].unsqueeze(0)     # 增加通道维度 [28,28]→[1,28,28]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)          
            
        return {'image':img,'label':label}

