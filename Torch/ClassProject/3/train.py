import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def plot_curve(data, title='Loss Curve'):
    plt.figure()
    plt.plot(data, color='blue')
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend(['value'])
    plt.show()

class SimpleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(96*96*3, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_binary_dataset():
    """创建二分类数据集: grey 和 yellow"""
    data_dir = './data'
    binary_dir = './data_binary'
    
    if os.path.exists(binary_dir):
        shutil.rmtree(binary_dir)
    
    for split in ['train', 'test']:
        for cls in ['grey', 'yellow']:
            os.makedirs(os.path.join(binary_dir, split, cls), exist_ok=True)
    
    for split in ['train', 'test']:
        for cls in ['grey', 'yellow']:
            src = os.path.join(data_dir, split, cls)
            dst = os.path.join(binary_dir, split, cls)
            for f in os.listdir(src):
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    
    print(f"Binary dataset created: grey + yellow")

def create_ternary_dataset():
    """创建三分类数据集: grey, yellow, blue"""
    data_dir = './data'
    ternary_dir = './data_ternary'
    
    if os.path.exists(ternary_dir):
        shutil.rmtree(ternary_dir)
    
    for split in ['train', 'test']:
        for cls in ['grey', 'yellow', 'blue']:
            os.makedirs(os.path.join(ternary_dir, split, cls), exist_ok=True)
    
    for split in ['train', 'test']:
        for cls in ['grey', 'yellow', 'blue']:
            src = os.path.join(data_dir, split, cls)
            dst = os.path.join(ternary_dir, split, cls)
            for f in os.listdir(src):
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    
    print(f"Ternary dataset created: grey + yellow + blue")

def train_binary():
    print("=" * 50)
    print("Task A: Binary Classification (Grey vs Yellow)")
    print("=" * 50)
    
    create_binary_dataset()
    
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    
    train_data = SimpleDataset('./data_binary/train', transform=transform)
    test_data = SimpleDataset('./data_binary/test', transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Classes: {train_data.classes}")
    
    net = Net(num_classes=2)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    train_loss = []
    for epoch in range(10):
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(x.size(0), -1)
            out = net(x)
            loss = criterion(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loss.append(loss.item())
            
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1)
            out = net(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    acc = correct / total
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    return acc, train_loss

def train_ternary():
    print("\n" + "=" * 50)
    print("Task B: Ternary Classification (Grey, Yellow, Blue)")
    print("=" * 50)
    
    create_ternary_dataset()
    
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])
    
    train_data = SimpleDataset('./data_ternary/train', transform=transform)
    test_data = SimpleDataset('./data_ternary/test', transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Classes: {train_data.classes}")
    
    net = Net(num_classes=3)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    train_loss = []
    for epoch in range(10):
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(x.size(0), -1)
            out = net(x)
            loss = criterion(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loss.append(loss.item())
            
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1)
            out = net(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    acc = correct / total
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    return acc, train_loss

if __name__ == '__main__':
    binary_acc, binary_loss = train_binary()
    ternary_acc, ternary_loss = train_ternary()
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Binary Classification Accuracy: {binary_acc*100:.2f}%")
    print(f"Ternary Classification Accuracy: {ternary_acc*100:.2f}%")
