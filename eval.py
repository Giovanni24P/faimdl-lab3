# Validation loop
from data.DataLoader import DataLoader
from models.CustomNet import CustomNet
from torch import nn
import torch


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            val_loss += loss.item()
            _, predicted = predictions.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy


def main():
    model = torch.load("./checkpoints/model1.t", weights_only=False)
    if torch.cuda.is_available():
        model = model.to_device('cuda')
    criterion = nn.CrossEntropyLoss()

    val_loader = DataLoader("somePath")
    print("Cardinality:", val_loader.getCardinality())
    val_loader = val_loader.getDataLoader()
    best_acc = 0

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)


    return

main()