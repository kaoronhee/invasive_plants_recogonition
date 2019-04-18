import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.utils.data
from torch.utils.data import DataLoader, distributed
import numpy as np
import time
from networks import*


# 定义本次训练版本号
exp_version = 'ias20190417'

# 定义随机旋转角度
rotation_degree = np.random.randint(15, 45)

# 定义学习率
lr = 0.005

# 定义隐藏层数
hidden_dim = 800

# 定义是否为断点继续学习
ifcontinue_train = False

# 定义模型训练保存位置
model_path = 'model/' + exp_version + '_params.pkl'

# 定义一般常量
train_acc = []
total_steps = 0  # 总训练次数
train_loss = []
test_accs=[]
test_loss=[]
epoch = 1
print_freq = 36


# 定义图片变换列表
mytransforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224,(0.8,1.2)),
        transforms.RandomRotation(rotation_degree),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ]
)


eval_transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
                                     transforms.ToTensor(),
                                     normalize])

train_data = datasets.ImageFolder('imgs/train', transform=mytransforms)
train_loader = DataLoader(train_data, batch_size=16,
                          shuffle=True)
eval_data = datasets.ImageFolder('imgs/eval', transform=transforms.Compose(
    [transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),transforms.ToTensor()]))
eval_loader=DataLoader(eval_data,batch_size=4,shuffle=True)

classes = train_data.classes

model = densenet201(pretrained=True)

model.classifier = nn.Linear(model.classifier.in_features, len(classes))

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=0)

# 如果有CUDA支撑，则模型推送到gpu运算
if torch.cuda.is_available():
    model.cuda()

# 如果是断点续练，则从保存的模型中加载训练好的模型
if ifcontinue_train:
    model.load_state_dict(torch.load(model_path))
    print('load saved model params from:', model_path)

# 训练和测试

for epoch in range(10):  # 全集训练次数
    if epoch > 3:
        lr = 0.001
    iteration = 0  # 分批训练次数
    start = time.time()
    batch_loss = 0
    tr_loss = 0
    train_total = 0
    train_correct = 0
    model.train()

    for batch, labels in train_loader:
        iteration += 1
        total_steps += 1
        images = Variable(batch)
        targets = Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        outputs = model(images)

        loss = loss_func(outputs, targets)
        batch_loss += loss
        tr_loss += loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        train_loss.append(loss.data.item())
        optimizer.step()

        train_total += labels.size(0)
        #_, predicted = torch.Tensor().max(outputs.data, 1)
        predicted = torch.argmax(outputs.data)
        predicted = predicted.cpu().numpy().astype(int)
        train_correct += (predicted == labels.numpy().astype(int)).sum()

        if total_steps % print_freq:
            print('Epoch:{0},Iteration:{1},LR:{2},Loss:{3:8.6f},{4:6.5f} sec/batch'.format(
                epoch, iteration, lr, batch_loss /
                print_freq, (time.time()-start)/print_freq
            ))
            start = time.time()
            batch_loss = 0

    train_acc = 100 * train_correct / train_total
    train_accs.append(train_acc)
    train_loss.append(tr_loss)
