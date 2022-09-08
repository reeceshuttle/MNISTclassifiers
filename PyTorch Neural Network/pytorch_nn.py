import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mnist import MNIST
import matplotlib.pyplot as plt
import time
torch.manual_seed(42)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# print(f"using {device} device")

class MNISTTraining(Dataset):
    """Contains the MNIST training dataset."""
    def __init__(self, mnist_location: str):
        mndata = MNIST(mnist_location)
        training_images, training_labels = mndata.load_training()
        assert len(training_images) == len(training_labels)
        self.training_images = training_images
        self.training_labels = training_labels

    def __len__(self):
        length = len(self.training_labels)
        assert length == 60000
        return length

    def __getitem__(self, index):
        image = torch.tensor(self.training_images[index], dtype=torch.float32)
        assert len(image) == 784
        label = torch.tensor(self.training_labels[index], dtype=torch.int64)
        return image, label

class MNISTTesting(Dataset):
    """Contains the MNIST testing dataset."""
    def __init__(self, mnist_location: str):
        mndata = MNIST(mnist_location)
        testing_images, testing_labels = mndata.load_testing()
        assert len(testing_images) == len(testing_labels)
        self.testing_images = testing_images
        self.testing_labels = testing_labels
    
    def __len__(self):
        length = len(self.testing_labels)
        assert length == 10000
        return length
    
    def __getitem__(self, index):
        image = torch.tensor(self.testing_images[index], dtype=torch.float32)
        assert len(image) == 784
        label = torch.tensor(self.testing_labels[index], dtype=torch.int64) #what dtype to use?
        return image, label

class MNISTNN(nn.Module):
    """Class for a FCNN designed to classify the MNIST dataset."""
    def __init__(self):
        super().__init__()
        dtype = torch.float32
        self.arch = nn.Sequential(nn.Linear(28*28, 32, dtype=dtype), nn.ReLU(), 
                                  nn.Linear(32, 24, dtype=dtype), nn.ReLU(), 
                                  nn.Linear(24, 10, dtype=dtype), nn.LogSoftmax(dim=1)) # they advice to use LogSoftmax when senting to NLLLoss, not Softmax
        
    def forward(self, input):
        logits = self.arch(input)
        return logits


def train(model: MNISTNN, training_data: DataLoader, loss_fn, optimizer):
    model.train()
    for (X, y) in training_data:
        X, y = X.to(device), y.to(device)
        # Forward pass:
        pred = model(X)
        loss = loss_fn(pred, y)
        # Back prop:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # for some reason this knows to refer to the correct model

def test(model: MNISTNN, testing_data: DataLoader, loss_fn):
    model.eval()
    num = len(testing_data)
    total_loss = 0
    correct_labels = 0
    for (x, y) in testing_data:
        pred = model(x)
        total_loss += loss_fn(pred, y)
        label = torch.argmax(pred).item()
        if label == y.item():
            correct_labels += 1
    print(f'average loss: {total_loss/num}')
    print(f'correct labels: {correct_labels} (out of {num})')
    return correct_labels/num, total_loss/num

def main():
    model = MNISTNN().to(device)
    print(model)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    dataloader_training = DataLoader(MNISTTraining('MNIST Dataset'), batch_size=5, shuffle=True)
    dataloader_testing = DataLoader(MNISTTesting('MNIST Dataset'), batch_size=1)
    dataloader_testing_of_training = DataLoader(MNISTTraining('MNIST Dataset'), batch_size=1)

    epochs = 1
    epoches = [i for i in range(epochs+1)]
    train_accuracy = []
    test_accuracy = []
    train_avgloss = []
    test_avgloss = []

    print('before:\n')
    print('test dataset:')
    acc, avgl = test(model, dataloader_testing, loss_fn)
    test_accuracy.append(acc)
    print('train dataset:')
    acc, avgl = test(model, dataloader_testing_of_training, loss_fn)
    train_accuracy.append(acc)
    print('training:\n')
    for epoch in range(epochs):
        start = time.time()
        print(f'epoch {epoch}:\n')
        train(model, dataloader_training, loss_fn, optimizer)
        print('test dataset:')
        acc, avgl = test(model, dataloader_testing, loss_fn)
        test_accuracy.append(acc)
        print('train dataset:')
        acc, avgl = test(model, dataloader_testing_of_training, loss_fn)
        train_accuracy.append(acc)
        print(f'epoch time: {time.time()-start}')
    plt.plot(epoches[1:], test_accuracy[1:], 'b', label='Testing')
    plt.plot(epoches[1:], train_accuracy[1:], 'r', label='Training')
    plt.legend()
    plt.title('Training vs Testing Accuracy')
    plt.show()
    # plt.savefig('pytorch_trainingvstesting.png', format='png')

    plt.plot(epoches, test_accuracy, 'b', label='Testing')
    plt.plot(epoches, train_accuracy, 'r', label='Training')
    plt.legend()
    plt.title('Training vs Testing Accuracy')
    plt.show()
    # plt.savefig('pytorch_trainingvstesting_withzero.png', format='png')

    # # saves model.
    # torch.save(model.state_dict(), 'pytorch_mnist_model') 
    # # reloads the saved model.
    # model2 = MNISTNN()
    # model2.load_state_dict(torch.load('pytorch_mnist_model'))
    # print('\n\nafter reloading:\n')
    # print('test dataset:')
    # test(model2, dataloader_testing, loss_fn)
    # print('train dataset:')
    # test(model2, dataloader_testing_of_training, loss_fn)
    
if __name__ == "__main__":
    main()
