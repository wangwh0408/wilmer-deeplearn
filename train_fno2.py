import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from fno2_model import FNO2d


class PoissonDataset(Dataset):
    def __init__(self, n_samples=1000, resolution=64):
        self.n_samples = n_samples
        self.resolution = resolution
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.n_samples):
            x, y, f, u = self._generate_poisson_problem()
            data.append((f, u))
        return data

    def _generate_poisson_problem(self):
        res = self.resolution
        x = np.linspace(0, 1, res)
        y = np.linspace(0, 1, res)
        X, Y = np.meshgrid(x, y)

        num_sources = np.random.randint(2, 6)
        f = np.zeros((res, res))
        for _ in range(num_sources):
            x0 = np.random.uniform(0.2, 0.8)
            y0 = np.random.uniform(0.2, 0.8)
            sigma = np.random.uniform(0.05, 0.15)
            amplitude = np.random.uniform(1.0, 5.0)
            f += amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

        u = self._solve_poisson_fft(f)
        return X, Y, f, u

    def _solve_poisson_fft(self, f):
        res = self.resolution
        kx = np.fft.fftfreq(res) * res
        ky = np.fft.fftfreq(res) * res
        KX, KY = np.meshgrid(kx, ky)

        f_hat = np.fft.fft2(f)
        u_hat = np.zeros_like(f_hat, dtype=np.complex128)

        mask = (KX**2 + KY**2) > 0
        u_hat[mask] = -f_hat[mask] / (4 * np.pi**2 * (KX[mask]**2 + KY[mask]**2))

        u = np.fft.ifft2(u_hat).real
        u = u - np.mean(u)
        u = (u - np.min(u)) / (np.max(u) - np.min(u))
        return u

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        f, u = self.data[idx]
        f_tensor = torch.tensor(f, dtype=torch.float32).unsqueeze(-1)
        u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(-1)
        return f_tensor, u_tensor


def train_fno2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    modes = 12
    width = 32
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    resolution = 64

    print('Generating dataset...')
    train_dataset = PoissonDataset(n_samples=800, resolution=resolution)
    test_dataset = PoissonDataset(n_samples=200, resolution=resolution)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Creating model...')
    model = FNO2d(modes, modes, width, in_channels=3, out_channels=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    print('Starting training...')
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

    print('Training completed!')

    print('Saving model...')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses,
        'test_loss': test_losses,
        'modes': modes,
        'width': width,
        'resolution': resolution
    }, 'fno2_model.pth')
    print('Model saved to fno2_model.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.savefig('training_loss.png')
    print('Loss curve saved to training_loss.png')

    return model, train_losses, test_losses


if __name__ == '__main__':
    train_fno2()
