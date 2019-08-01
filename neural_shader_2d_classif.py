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
        self.to_output = nn.Linear(2, 2)

    def forward(self, x):
        # pdb.set_trace()
        for layer in self.internal_layers:
            x = F.relu(x + layer(x))
        x = torch.log_softmax(self.to_output(x), dim = -1)

        return x

def SDFmap(p):
    return (torch.norm(p, dim = 1) >= 1).long()

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
    criterion    = nn.CrossEntropyLoss()
    # criterion    = nn.CrossEntropyLoss()
    optimizer    = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):
        train_loss    = 0
        train_correct = 0
        train_pred    = 0
        # pdb.set_trace()
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y   = X.to(device), y.to(device)
            y_pred = model(X)
            loss   = criterion(y_pred, y)
            # loss = (y_pred - y).abs().mean()
            loss.backward()
            optimizer.step()
            train_loss    += loss.item()
            train_pred    += len(y)
            train_correct += (y_pred.argmax(dim = -1) == y).sum().item()
        val_loss    = 0
        val_correct = 0
        val_pred    = 0
        for X, y in val_loader:
            X, y         = X.to(device), y.to(device)
            y_pred       = model(X)
            val_loss    += criterion(y_pred, y).item()
            val_pred    += len(y)
            val_correct += (y_pred.argmax(dim = -1) == y).sum().item()
        print(f'Epoch [{epoch + 1:4d} / {epochs:4d}] '
              f'Train loss: {train_loss / len(train_loader):6.3f}, '
              f'Train acc: {100*train_correct / train_pred:6.3f}%, '
              f'Val loss: {val_loss / len(val_loader):6.3f} '
              f'Train acc: {100*val_correct / val_pred:6.3f}%')

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
    print('float map(vec2 p) {')
    for i, layer in enumerate(model.internal_layers):
        values = [
            x.item() for x in layer.weight.transpose(0, 1).contiguous().view(-1)
        ]
        print(
            f'\tmat2 lin{i + 1:<3d}   = mat2(' +
            ', '.join(map(str, values)) +
            ');'
        )
        print(
            f'\tvec2 bias{i + 1:<3d}  = vec2(' +
            ', '.join(map(
                str,
                [bias_value.item() for bias_value in layer.bias]
            )) +
            ');'
        )
    values = [x.item() for x in model.to_output.weight.transpose(0, 1).contiguous().view(-1)]
    print(
        '\tmat2 to_out   = mat2(' +
        ', '.join(map(str, values)) +
        ');'
    )
    print(
        f'\tvec2 bias_out = vec2(' +
        ', '.join(map(
            str,
            [bias_value.item() for bias_value in model.to_output.bias]
        )) +
        ');'
    )
    print('\n')
    print('\tvec2 x = p;')
    for i in range(len(model.internal_layers)):
        print(f'\tx      = max(x + lin{i + 1:<3d} * x + bias{i + 1:<3d}, 0.);')
    print('\tx      = to_out * x + bias_out;')
    print()
    print('\treturn step(0.5, exp(x.y) / (exp(x.x) + exp(x.y)));\n}')

def main():
    device                     = torch.device('cuda')
    n_layers                   = 15
    dataset_size               = 20000
    epochs                     = 150
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
