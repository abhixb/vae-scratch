import torch
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 128
LR_RATE = 3e-4


def load_data():
    dataset = datasets.MNIST(
        root="dataset/",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataset, loader


def train(model, loader, optimizer, loss_fn):
    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(loader), desc=f"Epoch {epoch + 1}")
        for i, (x, _) in loop:
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)

            mu, sigma, x_reconstructed = model(x)  

            reconstruction_loss = loss_fn(x_reconstructed, x)

            kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            loss = reconstruction_loss + kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

def get_digit_encodings(model, dataset):
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, INPUT_DIM))
        encodings.append((mu, sigma))

    return encodings


def inference(digit, encodings, model, num_examples=1):
    mu, sigma = encodings[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")


def main():
    dataset, loader = load_data()

    model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    train(model, loader, optimizer, loss_fn)

    encodings = get_digit_encodings(model, dataset)
    for digit in range(10):
        inference(digit, encodings, model, num_examples=5)


if __name__ == "__main__":
    main()