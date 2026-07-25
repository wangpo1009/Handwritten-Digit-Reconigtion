import torch
from torch import nn

class PoModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Out: 32, 14, 14

            nn.Conv2d(
                in_channels= 32,
                out_channels = 64,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            #Out 64, 7, 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), 
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x) 
        return x

# Train loops
def train_step(model, dataloader, lossfn, optimizer, device):
    model.train()
    train_loss = 0
    train_acc = 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        #forward
        y_logits = model(X)

        loss = lossfn(y_logits, y)

        #backprop
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        y_pred = torch.argmax(y_logits, dim=1)

        train_acc += (y_pred == y).sum().item() / len(y)

    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(model,
              dataloader,
              loss_fn,
              device):

    model.eval()

    test_loss = 0
    test_acc = 0

    with torch.inference_mode():

        for X, y in dataloader:

            X = X.to(device)
            y = y.to(device)

            y_logits = model(X)

            loss = loss_fn(y_logits, y)

            test_loss += loss.item()

            y_pred = torch.argmax(y_logits, dim=1)

            test_acc += (y_pred == y).sum().item() / len(y)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc




