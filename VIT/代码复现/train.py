import torch
import torch.nn as nn
from torch import optim
import timeit
from tqdm import tqdm
from utils import get_loaders
from model import Vit

device = 'cuda'
EPOCHS = 50

BATCH_SIZE =16
TRAIN_DIR = './'
TEST_DIR = './'

IN_CHANNELS = 1
IMG_SIZE = 28
PATCH_SIZE = 4
EMBED_DIM = (PATCH_SIZE ** 2)*IN_CHANNELS # 这里考虑到输入图片尺度较小，不采用base模型中的768
DROPOUT = 0.001

NUM_HEADS =8
ACTIVATION = "gelu"
NUM_ENCODERS = 12
NUM_CLASSES = 10

LEARNNING_RATE = 1e-4
ADAM_WEIGHT_DECAY = 1e-2
ADAM_BEATS = (0.9,0.999)


train_loader,valid_loader,test_loader = get_loaders(TRAIN_DIR,TEST_DIR,BATCH_SIZE)
model = Vit(IN_CHANNELS,PATCH_SIZE,EMBED_DIM,IMG_SIZE,NUM_HEADS,ACTIVATION,DROPOUT,NUM_ENCODERS,NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),betas=ADAM_BEATS,lr = LEARNNING_RATE,weight_decay=ADAM_WEIGHT_DECAY)

if __name__ == '__main__':
    print(len(train_loader.dataset))
    print(len(valid_loader.dataset))
    print(len(test_loader.dataset))
    start = timeit.default_timer()

    for epoch in range(EPOCHS):
        model.train()
        train_labels = []
        train_preds = []
        train_running_loss = 0

        for idx,train_images in enumerate(tqdm(train_loader,position=0,leave=True)):
            img = train_images["image"].float().to(device)
            label = train_images['label'].type(torch.uint8).to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred,dim=1)

            train_labels.extend(label.cpu().detach())
            train_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred,label) # 默认情况下loss是标量

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
        train_loss = train_running_loss / len(train_loader) # 每个样本平均损失

        model.eval()
        val_labels = []
        val_preds = []
        val_running_loss = 0
        with torch.no_grad():
            for idx,val_images in enumerate(tqdm(valid_loader,position=0,leave=True)):
                img = val_images["image"].float().to(device)
                label = val_images['label'].type(torch.uint8).to(device)
                y_pred = model(img)
                y_pred_label = torch.argmax(y_pred,dim=1)
                val_labels.extend(label.cpu().detach())
                val_preds.extend(y_pred_label.cpu().detach())
                loss = criterion(y_pred,label) 
                val_running_loss += loss.item()
        val_loss = val_running_loss / len(valid_loader) # 每个样本平均损失

        print('-'* 30)
        print(f'Train Loss Epoch {epoch + 1} : {train_loss:.4f}')
        print(f'val Loss Epoch {epoch + 1} : {val_loss:.4f}')
        print(f"Train Acc Epoch {epoch + 1} : {sum(1 for x,y in zip(train_preds,train_labels) if x==y)/len(train_labels):.4f}")
        print(f"valid Acc Epoch {epoch + 1} : {sum(1 for x,y in zip(val_preds,val_labels) if x==y)/len(val_labels):.4f}")
        print('-'* 30)
    stop = timeit.default_timer()
    print(f"trainning time: {stop - start:.2f}s")


    model.eval()
    test_labels = []
    test_preds = []
    test_running_loss = 0
    with torch.no_grad():
        for idx,test_images in enumerate(tqdm(test_loader,position=0,leave=True)):
            img = test_images["image"].float().to(device)
            label = test_images['label'].type(torch.uint8).to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred,dim=1)
            test_labels.extend(label.cpu().detach())
            test_preds.extend(y_pred_label.cpu().detach())
            loss = criterion(y_pred,label) 
            test_running_loss += loss.item()
    test_loss = val_running_loss / len(test_loader.dataset) 
    print('-'* 30)
    print(f'Test Loss Epoch {epoch + 1} : {test_loss:.4f}')
    print(f"Test Acc Epoch {epoch + 1} : {sum(1 for x,y in zip(test_preds,test_labels) if x==y)/len(test_preds):.4f}")
    print('-'* 30)