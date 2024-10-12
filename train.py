import torch
from torch import optim
from torch.xpu import device

from models import MyModel
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange, tqdm
from genData import get_dataloader


def train(epochs=100, device="cuda:0"):
    model = MyModel(1, [16, 64, 32, 16], 1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    scheduler = ExponentialLR(optimizer, gamma=0.95)
    loss_fn = F.mse_loss

    model.train()
    for e in range(epochs):
        total_loss = 0
        for x, y in tqdm(get_dataloader(102400, mode="train"), desc=f"Epoch [{e+1}/{epochs}]"):
            x = x.view(-1, 1).to(device)
            y = y.view(-1, 1).to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        # tqdm.write("epoch: {}, loss: {}".format(epoch+1, total_loss / 1000))

    torch.save(model, "./results/model.pt")


if __name__ == "__main__":
    train()