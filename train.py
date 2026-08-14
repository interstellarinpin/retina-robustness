import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# print(len(training_data))
# print(len(test_data))

batch_size = 64

train_dataloader = DataLoader(
    training_data,
    batch_size=batch_size
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size
)

train_features, train_labels = next(iter(train_dataloader))

# print(train_features.shape)
# print(train_labels.shape)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits

model = NeuralNetwork()
X, y = next(iter(train_dataloader))

pred = model(X)

# print(X.shape)
# print(pred.shape)
# print(pred[0])
# print(pred[0].argmax())
# print(y[0])

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-3

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

# def train(dataloader, model, loss_fn, optimizer):
#     model.train()

#     for X, y in dataloader:
#         # 1. Forward pass
#         pred = model(X)

#         # 2. Calculate average loss for this batch
#         loss = loss_fn(pred, y)

#         # 3. Calculate gradients
#         loss.backward()

#         # 4. Update every parameter
#         optimizer.step()

#         # 5. Clear gradients before next batch
#         optimizer.zero_grad()
    
def train(dataloader, model, loss_fn, optimizer):
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"Batch {batch}: loss = {loss.item():.4f}")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)

    model.eval()
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size

    print(f"Test accuracy: {100 * correct:.1f}%")

print(model)

epochs = 5

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)