import torch
from scipy.ndimage import label

from genData import get_dataloader
import matplotlib.pyplot as plt


def draw_chart(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=1, label="y vs x")
    plt.title("y vs x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()



def test():
    model = torch.load("./results/model.pt").to("cpu")
    model.eval()

    x_vals = []
    y_vals = []
    y_pred_vals = []

    with torch.no_grad():
        for x, y in get_dataloader(10000, mode="test"):
            x = x.view(-1, 1)
            y = y.view(-1, 1)
            y_pred = model(x)
            x_vals.extend(x.detach().numpy().flatten())
            y_vals.extend(y.detach().numpy().flatten())
            y_pred_vals.extend(y_pred.detach().numpy().flatten())

    # draw a chart
    draw_chart(x_vals, y_pred_vals)
    draw_chart(x_vals, y_vals)


if __name__ == "__main__":
    test()