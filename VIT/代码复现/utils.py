from dataset import MNISTDataset
from torch.utils.data import DataLoader
from torchvision import datasets
import random
from torch.utils.data import DataLoader


def get_loaders(train_dir,test_dir,batch_size):
    raw_train = datasets.MNIST(
        root=train_dir, 
        train=True,
        download=True,
        transform=None  # 禁用默认转换以自定义处理
    )
    raw_test = datasets.MNIST(
        root=test_dir, 
        train=False,
        download=True
    )
   # 计算训练集和验证集的大小
    train_size = int(0.8 * len(raw_train))
    indices = list(range(len(raw_train)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    # 根据索引获取训练和验证数据
    train_data = raw_train.data[train_indices]
    train_targets = raw_train.targets[train_indices]
    val_data = raw_train.data[val_indices]
    val_targets = raw_train.targets[val_indices]
    class DatasetWithAttributes:
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets
    train_dataset_obj = DatasetWithAttributes(train_data, train_targets)
    val_dataset_obj = DatasetWithAttributes(val_data, val_targets)

    train_dataset = MNISTDataset(train_dataset_obj, augment=True) 
    valid_dataset = MNISTDataset(val_dataset_obj, augment=False) 
    test_dataset = MNISTDataset(raw_test, augment=False)         
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    return train_loader,valid_loader,test_loader


    
