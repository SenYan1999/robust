import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from main import get_model, args
from tqdm import tqdm

def get_dataloader(dataset, half, batch_size, workers, distributed=False):
    mean = {
        'CIFAR10': (0.4914, 0.4822, 0.4465),
        'CIFAR100': (0.5071, 0.4867, 0.4408),
        'STL10': (0.5, 0.5, 0.5),
        'IMAGENET': (0.485, 0.456, 0.406)
    }

    std = {
        'CIFAR10': (0.2023, 0.1994, 0.2010),
        'CIFAR100': (0.2675, 0.2565, 0.2761),
        'STL10': (0.5, 0.5, 0.5),
        'IMAGENET': (0.229, 0.224, 0.225)
    }
    normalize = transforms.Normalize(mean=mean[dataset],
                                     std=std[dataset])

    if dataset.startswith('CIFAR'):
        train_dataset = datasets.__dict__[dataset](root='../kd_mixup/src/data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
        if half:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       num_workers=workers, pin_memory=True, sampler=train_sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       num_workers=workers, pin_memory=True, shuffle=True)

        val_dataset = datasets.__dict__[dataset](root='../kd_mixup/src/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=128, shuffle=False,
                                                 num_workers=workers, pin_memory=True)

    return train_loader, val_loader

def evaluate_distance(text_file, text_model):
    # evaluate if CV distance is smaller than NLP thus the small need to be relative
    # image_loader, _ = get_dataloader('CIFAR10', False, 128, 4, False)
    # stack = []
    # for batch in image_loader:
    #     x, y = map(lambda x: x.cuda(), batch)
    #     stack.append(x)
    # image_mean = torch.zeros(x.shape[1:]).cuda()
    # for i in stack:
    #     for j in i:
    #         image_mean += j
    # image_mean = image_mean / len(stack) / stack[0].shape[0]
    
    # stack = []
    # for batch in image_loader:
    #     x, y = map(lambda x: x.cuda(), batch)
    #     for i in x:
    #         stack.append(torch.sqrt(torch.sum((i - image_mean) ** 2)).item())
    # image_dist = np.mean(stack)
    # print('image dist: %.2f' % image_dist)

    text_data = torch.load(text_file)
    try:
        n_vocab = len(text_data.word2idx)
    except:
        n_vocab = None
    model = get_model(args, n_vocab)
    model.load_state_dict(torch.load(text_model)['model_state'])
    model = model.cuda()
    stack = []
    text_loader = torch.utils.data.DataLoader(text_data, batch_size = 64)
    for batch in tqdm(text_loader):
        batch = list(map(lambda x: x.cuda(), batch))
        with torch.no_grad():
            _, embed = model(batch)
        stack.append(embed)
    text_mean = torch.zeros(embed.shape[1:]).cuda()
    for i in stack:
        for j in i:
            text_mean += j

    dist_stack = []
    for i in stack:
        for j in i:
            dist_stack.append(torch.sqrt(torch.sum((j - text_mean) ** 2)).item())
    text_dist = np.mean(dist_stack)
    print('text dist: %.2f' % text_dist)

if __name__ == '__main__':
    evaluate_distance('glue_data/CoLA/no_bert_bert-base-uncased_train.pt', 'save_model/BiLSTMModel.pt')
