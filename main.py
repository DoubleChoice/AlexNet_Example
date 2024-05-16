from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split,DataLoader
import torch
import torch.nn as nn
import model

torch.device('cuda:0')

transform = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ]
)
"""
图片数据集：
图片文件夹 /root/class_1/xxx.jpg
              |
              /class_2/xxx.jpg
              |
              /...
图片 -> ImagerFolder() -> dataset -> random_split() -> train_dataset and vali_dataset -> DataLoader() -> train_dataloader and vali_dataloader
"""
dataset = ImageFolder('/',transform=transform)
train_len = int(len(dataset) * 0.7)
vali_len = int(len(dataset) * 0.3)
train_dataset , vali_dataset = random_split(dataset=dataset,lengths=[train_len,vali_len],generator=torch.Generator().manual_seed(0))
train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=0)
vali_dataloader = DataLoader(vali_dataset,batch_size=32,shuffle=True,num_workers=0)
print(dataset.classes)
print(dataset[0][0].size())
###########################
net = model.model()
#net = torch.load('model_pth/net_9.pth')    载入模型超参数
loss_fun = nn.CrossEntropyLoss()    # 损失函数loss_fun 使用交叉熵损失 or other -> loss = loss_fun(模型输出的预测，实际标签) -> backward()反向传播
optimizer = torch.optim.Adam(net.parameters(),lr= 0.0001) # 优化器optimizer 使用 adam or other 优化器(模型参数 , 学习率) -> optimizer.zero_grad() -> optimizer.step()


"""
模型训练与验证部分：
for epoch in range(n) 定义n个训练epoch
    训练：
    for datas,labels in train_dataloader:
        loss_fun
        optimizer
        metrics : loss
        ...
    每轮训练总次数 = 训练数据集size / batch_size
        
    验证：
    for datas,labels in vali_dataloader：
        metrics : acc...
        ...
    每轮验证总次数 = 验证数据集size / batch_size
"""
train_step = 0
vali_step = 0
for epoch in range(10):
    loss = 0.0
    total_loss = 0
    print("epoch:{}".format(epoch))
    for datas,labels in train_dataloader:
            optimizer.zero_grad()
            outputs = net(datas)
            loss = loss_fun(outputs,labels)
            loss.backward()
            optimizer.step()
            train_step +=1
            total_loss += loss.item()
            if train_step % 10 == 0:
                print("训练次数：{},train_loss : {}".format(train_step,loss.item()))
    total_loss /= len(train_dataset)/train_dataloader.batch_size
    print("every epoch total_loss:{}".format(total_loss))

    acc = 0
    total_acc = 0 #每一个epoch内的平均acc
    with torch.no_grad():
        for datas,labels in vali_dataloader:
            outputs = net(datas) # [batch_size ,classes_size] -> [32,5]   32个样本的每行五个经过softmax后的实数
            loss = loss_fun(outputs,labels)
            #print(torch.max(outputs,dim=1)) #输出32行中五列中最大的实数
            pred = torch.max(outputs, dim=1)[1]
            #print(torch.max(outputs, dim=1)[1])  #输出最大的实数对应的标签\
            acc = (pred == labels).sum().item() / len(labels)
            total_acc += acc
            vali_step += 1
            if vali_step % 10 == 0:
                print("验证次数：{},acc：{}".format(vali_step,acc))
        total_acc /= len(vali_dataset) / vali_dataloader.batch_size
        print("every epoch total_acc {}".format(total_acc))

    torch.save(net,"model_pth/net_{}.pth".format(epoch)) #写入模型文件