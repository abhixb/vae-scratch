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
IMAGE_SIZE = 64
INPUT_DIM = 3 * IMAGE_SIZE * IMAGE_SIZE
H_DIM = 512
Z_DIM = 128
NUM_EPOCHS = 10
BATCH_SIZE = 128
LR_RATE = 3e-4


from torch.utils.data import Dataset
from PIL import Image
import os

class CelebADataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def load_data():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    full_dataset = CelebADataset("dataset/celeba/img_align_celeba", transform=transform)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return full_dataset, train_loader, val_loader

def train(model, train_loader, val_loader, optimizer, loss_fn):
    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1} [Train]", total=len(train_loader))
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

        val_loss = evaluate(model, val_loader, loss_fn, epoch)
        print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.2f}")
        save_reconstructions(model, val_loader, epoch)


def evaluate(model, val_loader, loss_fn, epoch):
    model.eval()
    total_loss = 0
    loop = tqdm(enumerate(val_loader), desc=f"Epoch {epoch + 1} [Val]", total=len(val_loader))
    with torch.no_grad():
        for i, (x, _) in loop:
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            mu, sigma, x_reconstructed = model(x)

            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = reconstruction_loss + kl_div

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    return total_loss / len(val_loader)


def save_reconstructions(model, val_loader, epoch):
    model.eval()
    x, _ = next(iter(val_loader))
    x = x[:16].to(DEVICE)
    with torch.no_grad():
        _, _, x_reconstructed = model(x.view(x.shape[0], INPUT_DIM))
    x_reconstructed = x_reconstructed.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
    combined = torch.cat([x, x_reconstructed])
    save_image(combined, f"reconstructions_epoch{epoch + 1}.png", nrow=16)


def inference(model, num_samples=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, Z_DIM).to(DEVICE)
        samples = model.decode(z).view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
    save_image(samples, "generated_samples.png", nrow=4)


def main():
    train_dataset, train_loader, val_loader = load_data()

    model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    train(model, train_loader, val_loader, optimizer, loss_fn)
    inference(model, num_samples=16)
    torch.save(model.state_dict(), "vae_celeba.pth")


if __name__ == "__main__":
    main()