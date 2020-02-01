import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from kernel_conv.conv import kernel_wrapper
from kernel_conv.kernels import *
from tqdm import tqdm

def main():
    # Parsing command line args
    parser = argparse.ArgumentParser(description='CIFAR10 example')
    parser.add_argument('--kernel', type=str, default=None,
                        help='Kernel type to use: [gaussian, polynomial, sigmoid] (default: None)')

    parser.add_argument('--epoch', type=int, default=2, help='Number of epochs (default: 2)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch suze (default: 4)')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU? (default: True)')

    args = parser.parse_args()

    device = 'cpu'
    if args.gpu:
        device = 'cuda'

    # Initiating network
    resnet50 = torchvision.models.resnet50()
    resnet50._modules['fc'] = torch.nn.Linear(2048, 10, True)

    if args.kernel == 'gaussian':
        kernel_wrapper(resnet50, GaussianKernel())
    elif args.kernel == 'polynomial':
        kernel_wrapper(resnet50, PolynomialKernel())
    elif args.kernel == 'sigmoid':
        kernel_wrapper(resnet50, SigmoidKernel())
    elif args.kernel is not None:
        raise Exception('Invalid kernel')

    resnet50.to(device)

    # Loading datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet50.parameters(), lr=0.001, momentum=0)

    print('=' * 5 + 'TRAINING' + '=' * 5)
    for epoch in tqdm(range(args.epoch)):
        running_loss = 0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optmizer.zero_grad()
            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.items()
        print('Epoch %i complete, loss: %f' % (epoch, running_loss))

    print('=' * 5 + 'TESTING' + '=' * 5)
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

if __name__ == '__main__':
    main()
