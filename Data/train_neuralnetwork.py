import torch
from Cifadataset import MyDataset
from model import SimpleNeuralNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report

if __name__ == '__main__':
    num_epochs = 5
    train_dataset = MyDataset(root='D:\Code Pytorch\Data\cifa-10-data\cifar-10-batches-py',train=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataset = MyDataset(root='D:\Code Pytorch\Data\cifa-10-data\cifar-10-batches-py',train=False)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )

    model = SimpleNeuralNetwork(num_class = 10)
    criterion =  nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    num_iters = len(train_dataloader)
    for epoch in range(num_epochs):
        model.train()
        for iter,(images, labels) in enumerate(train_dataloader):
            # forward
            outputs = model(images) # forward
            loss_value = criterion(outputs, labels)
         #   print("Epoch {}/{}. Iteration {}/{}. Loss {}".format(epoch+1, num_epochs,iter+1,num_iters,loss_value))

            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions, dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)

        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        print("Epochs {}".format(epoch+1))
        print(classification_report(all_labels, all_predictions))


