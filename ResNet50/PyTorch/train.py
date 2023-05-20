# -*- coding:utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import dataset
# from resnet50 import ResNet50
from concise_resnet50 import ResNet50

input_size = 224
classes_num = 5
class_name2index = dataset.class_name2index
index2class_name = dataset.index2class_name
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(int(input_size * 1.143)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

torch.manual_seed(7)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    step_total = len(train_loader)
    loss_epoch = 0
    correct_epoch = 0
    for step, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, label)
        pred = output.argmax(dim=1)
        correct = pred.eq(label).sum().item()
        acc = 100. * correct / len(label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print("Train Epoch:[{}/{}], Step:[{}/{}], Loss:{:.4f}, Accuracy:{:.2f}%, Lr:{:.5f}".format(
                epoch, args.epochs, step + 1, step_total, loss.item(), acc, optimizer.param_groups[0]["lr"]
            ))

        loss_epoch += loss.item()
        correct_epoch += correct

    # writer.add_graph(model, image)

    train_loss = loss_epoch / len(train_loader)
    train_acc = 100. * correct_epoch / len(train_loader.dataset)

    print("Train set: Loss:{:.4f}, Accuracy:{:.2f}%".format(
        train_loss, train_acc
    ))

    return train_loss, train_acc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            test_loss += nn.CrossEntropyLoss(reduction="sum")(output, label).item()
            # pred = output.argmax(dim=1, keepdim=True)
            pred = output.argmax(dim=1)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print("Test set: Loss:{:.4f}, Accuracy:{:.2f}%".format(
        test_loss, acc
    ))

    return test_loss, acc


def run():
    # 获取数据集目录
    data_root = args.data_dir

    # 创建dataset
    train_dataset = dataset.FlowerDataset(
        images_dir=os.path.join(data_root, "train"),
        transform=data_transform["train"]
    )
    val_dataset = dataset.FlowerDataset(
        images_dir=os.path.join(data_root, "val"),
        transform=data_transform["val"]
    )

    kwargs = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 构建网络
    model = ResNet50(classes_num=classes_num)
    # 加载预训练模型
    if os.path.isfile(args.pre_model_dir):
        print("Loading pretrained parameters...")
        pretrained_dict = torch.load(args.pre_model_dir, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "fc" not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-9)

    print("Start training!")
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        val_loss, val_acc = test(model, device, val_loader)
        scheduler.step()

        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)

        if args.save_interval != -1 and epoch % args.save_interval == 0:
            save_path = os.path.join(args.model_save_dir, "ResNet50_%s.pth" % epoch)
            torch.save(model.state_dict(), save_path)

        if val_acc > best_acc:
            best_save_path = os.path.join(args.model_save_dir, "best.pth")
            torch.save(model.state_dict(), best_save_path)
            best_acc = val_acc
            print("Succeeded saving best.pth, and val accuracy is %.2f%%" % best_acc)
        print("Previous best accuracy is : %.2f%%" % best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch ResNet50")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--pre-model-dir", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="../../flower_classify_dataset/")
    parser.add_argument("--model-save-dir", type=str, default="./model/")
    parser.add_argument("--save-interval", type=int, default=10)
    args = parser.parse_args()
    print(args)

    writer = SummaryWriter(args.model_save_dir)

    run()
