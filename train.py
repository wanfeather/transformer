import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ViT

def save_model(model, num):
    print('Saving model...')
    torch.save(model.module.state_dict(), 'model/model_{}.pth'.format(num))
    print('Model {} saved\n'.format(num))

def check_accuracy(loader, model, compute_deivce):
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(compute_deivce)
            y = y.to(compute_deivce)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        accuracy = float(num_correct)/float(num_samples)*100.0
        
        # print('Got {} / {} with accuracy {}%'.format(num_correct, num_samples, accuracy))

    return accuracy

class MyTranspose:
    def __call__(self, x):
        x = x.view(x.size(0), 1, x.size(-1) ** 2)
        x = x.permute(1, 0, 2)

        return x

def main():
    batch_size = 4096
    num_epochs = 20
    lr = 1e-3
    device = torch.device('cuda')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
        MyTranspose()
    ])

    train_dataset = datasets.MNIST(root = './data', train = True, transform = transform)
    test_dataset = datasets.MNIST(root = './data', train = False, transform = transform)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)

    model = ViT(1, 784, 8, 1000, 10).to(device)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    model = torch.nn.DataParallel(model, device_ids = [0, 1])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    with open('result.csv', 'w') as f:
        for epoch in range(num_epochs):
            losses = []

            model.train()
            for data, target in train_dataloader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                target_pre = model(data)
                loss = criterion(target_pre, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            training_loss = sum(losses)/len(losses)

            losses = []
            model.eval()
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)

                target_pre = model(data)
                loss = criterion(target_pre, target)
                losses.append(loss.item())
            testing_loss = sum(losses)/len(losses)

            train_acc = check_accuracy(train_dataloader, model, device)
            test_acc = check_accuracy(test_dataloader, model, device)

            print('training loss:{}\ntesting loss:{}\ntraining accuracy:{}\ntesting accuracy:{}\n'.format(training_loss, testing_loss, train_acc, test_acc))
            f.write('{},{},{},{},{}\n'.format(epoch, training_loss, testing_loss, train_acc, test_acc))

            save_model(model, epoch)

if __name__ == '__main__':
    main()
