import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader

class SDFModel(nn.Module):
    def __init__(self, n_layers):
        super(SDFModel, self).__init__()
        self.internal_layers = nn.Sequential(*[nn.Linear(2, 2) for _ in range(n_layers)])
        self.to_output = nn.Linear(2, 1)

    def forward(self, x):
        # pdb.set_trace()
        for layer in self.internal_layers:
            x = F.relu(x + layer(x))
        x = torch.sigmoid(self.to_output(x))

        return x

def SDFmap(p):
    return (torch.norm(p, dim = 1) >= 1).float()

def create_dataset(n_samples, mini, maxi):
    X = torch.empty(n_samples, 2).uniform_(mini, maxi)
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
    # criterion    = nn.CrossEntropyLoss()
    optimizer    = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        # pdb.set_trace()
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y   = X.to(device), y.to(device)
            y_pred = model(X)
            loss   = criterion(y_pred, y)
            # loss = (y_pred - y).abs().mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss = 0
        for X, y in val_loader:
            X, y      = X.to(device), y.to(device)
            y_pred    = model(X)
            val_loss += criterion(y_pred, y).item()
        print(f'Epoch [{epoch + 1} / {epochs}]'
              f'Train loss: {train_loss / len(train_loader)}, '
              f'Val loss: {val_loss / len(val_loader)}')

    evaluate(
        model         = model,
        val_dataset   = val_dataset,
        batch_size    = 128,
        device        = device
    )
def evaluate(model, val_dataset, batch_size, device, n_samples = 5):
    val_loader   = DataLoader(
        dataset    = val_dataset,
        batch_size = batch_size,
        shuffle    = False
    )
    for X, y in val_loader:
        X, y      = X.to(device), y.to(device)
        y_pred    = model(X)
        for i, (X_, y_, y_pred_) in enumerate(zip(X.cpu(), y.cpu(), y_pred.cpu())):
            if i == n_samples:
                return
            print(X_, y_, y_pred_)


def generate_map_function(model):
    print(
        '''float sigmoid(float x) {
\treturn 1 / (1 + exp(-x));
}\n''')

    print('float map(vec2 p) {')
    for i, layer in enumerate(model.internal_layers):
        values = [
            x.item() for x in layer.weight.transpose(0, 1).contiguous().view(-1)
        ]
        print(
            f'\tmat2 lin{i + 1}      = mat2(' +
            ', '.join(map(str, values)) +
            ');'
        )
        print(
            f'\tvec2 bias{i + 1}     = vec2(' +
            ', '.join(map(
                str,
                [bias_value.item() for bias_value in layer.bias]
            )) +
            ');'
        )
    values = [x.item() for x in model.to_output.weight.transpose(0, 1).contiguous().view(-1)]
    print(
        '\tvec2 to_out    = vec2(' +
        ', '.join(map(str, values)) +
        ');'
    )
    print(
        f'\tfloat bias_out = {model.to_output.bias.item()};'
    )
    print('\tvec2 x = p;')
    for i in range(len(model.internal_layers)):
        print(f'\tx = max(lin{i + 1} * x + bias{i + 1}, 0.);')
    print('\tfloat d = sigmoid(dot(to_out, x) + bias_out);')
    print()
    print('\treturn d;\n}')

def main():
    device                     = torch.device('cuda')
    n_layers                   = 1200
    dataset_size               = 10000
    epochs                     = 100
    batch_size                 = 512
    val_prop                   = .2
    dataset                    = create_dataset(dataset_size, -1, 1)
    val_size                   = round(val_prop * dataset_size)
    print(dataset_size, val_size)
    train_dataset, val_dataset = random_split(
        dataset,
        [dataset_size - val_size, val_size]
    )
    model                      = SDFModel(
        n_layers = n_layers
    )
    model                      = model.to(device)
    train(
        model         = model,
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
        epochs        = epochs,
        batch_size    = batch_size,
        device        = device
    )
    evaluate(
        model         = model,
        val_dataset   = val_dataset,
        batch_size    = batch_size,
        device        = device
    )
    generate_map_function(model)

if __name__ == '__main__':
    main()
