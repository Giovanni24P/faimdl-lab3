from data.DataLoader import DataLoader
from models.CustomNet import CustomNet
from torch import nn
import torch
from eval import validate

def train(epoch, model:nn.Module, train_loader, criterion:nn.CrossEntropyLoss, optimizer:torch.optim.SGD):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # forward pass
        predictions = model(inputs)

        loss = criterion(predictions, targets)

        running_loss += loss.item()
        _, predicted = predictions.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Backpropagation
        model.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')


def main():
    train_path = "xxxPATHxxx"
    val_path = "xxxPATH2xxxx"

    if torch.cuda.is_available():
        model = CustomNet().cuda()
    else:
        model = CustomNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_loader = DataLoader(train_path)
    print("Train cardinality:", train_loader.getCardinality())
    train_loader = train_loader.getDataLoader()

    val_loader = DataLoader(val_path)
    print("Validation cardinality:", val_loader.getCardinality())
    val_loader = val_loader.getDataLoader()


    best_acc = 0

    # Run the training process for {num_epochs} epochs
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        val_accuracy = validate(model, val_loader, criterion)
        best_acc = max(best_acc, val_accuracy)
    return


main()