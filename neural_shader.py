import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader

class SDFModel(nn.Module):
    def __init__(self, n_layers):
        super(SDFModel, self).__init__()
        self.internal_layers = nn.Sequential(*[nn.Linear(3, 3, bias = False) for _ in range(n_layers)])
        self.to_output = nn.Linear(3, 1, bias = False)


    def forward(self, x):
        for layer in self.internal_layers:
            x = F.relu(layer(x))
        x = self.to_output(x)

        return x

def SDFmap(p):
    return torch.norm((p - torch.tensor([0., 0., 5.])), dim = 1) - 1.

def create_dataset(n_samples, mini, maxi):
    X = torch.empty(n_samples, 3).uniform_(mini, maxi)
    y = SDFmap(X)

    return TensorDataset(X, y)

def train(model, train_dataset, val_dataset, epochs, batch_size, device):
    train_loader = DataLoader(
        dataset    = train_dataset,
        batch_size = batch_size,
        shuffle    = True
    )
    val_loader   = DataLoader(
        dataset    = val_dataset,
        batch_size = batch_size,
        shuffle    = False
    )
    criterion    = nn.MSELoss()
    optimizer    = optim.Adam(model.parameters())

    for epoch in range(epochs):
        train_loss = 0
        for X, y in train_loader:
            X, y   = X.to(device), y.to(device)
            y_pred = model(X)
            loss   = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        val_loss = 0
        for X, y in val_loader:
            X, y      = X.to(device), y.to(device)
            y_pred    = model(X)
            val_loss += criterion(y_pred, y).item()
        print(f'Epoch [{epoch + 1} / {epochs}]'
              f'Train loss: {train_loss / len(train_loader)}, '
              f'Val loss: {val_loss / len(val_loader)}')

def generate_map_function(model):
    print('float map(vec3 p) {')
    for i, layer in enumerate(model.internal_layers):
        values = [
            x.item() for x in layer.weight.view(-1)
        ]
        print(
            f'\tmat3 lin{i + 1} = mat3(' +
            ', '.join(map(str, values)) +
            ');'
        )
    values = [x.item() for x in model.to_output.weight.view(-1)]
    print(
        '\tvec3 to_out = vec3(' +
        ', '.join(map(str, values)) +
        ');'
    )
    print('\tvec3 x = p;')
    for i in range(len(model.internal_layers)):
        print(f'\tx = max(lin{i + 1} * x, 0.);')
    print('\tfloat d = dot(to_out, x);')
    print()
    print('\treturn d;\n}')

def main():
    device                     = torch.device('cuda')
    dataset_size               = 5000
    epochs                     = 50
    batch_size                 = 128
    val_prop                   = .2
    dataset                    = create_dataset(dataset_size, -10, 10)
    val_size                   = round(val_prop * dataset_size)
    print(dataset_size, val_size)
    train_dataset, val_dataset = random_split(
        dataset,
        [dataset_size - val_size, val_size]
    )
    model                      = SDFModel(
        n_layers = 5
    )
    model                      = model.to(device)
    train(
        model         = model,
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
        epochs        = epochs,
        batch_size    = 128,
        device        = device
    )
    generate_map_function(model)

if __name__ == '__main__':
    main()
