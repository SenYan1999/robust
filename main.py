import os
import torch
import sys

import torch.multiprocessing as mp
import torch.distributed as dist

from args import args
from utils import Trainer, GlueDataset, init_logger, SNLIDataset, BertDataset, NormalDataset
from apex import amp
from model import Bert, PGD, RandomBert, SmartPerturbation, RandomPretrainBert, PretrainBert, BiLSTMModel
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

def get_model(args, n_vocab=None):
    if args.bert_type == 'no_bert':
        model = BiLSTMModel(args.lstm_d_embed, args.lstm_d_hidden, n_vocab, p_drop=args.drop_p)
    else:
        model = Bert(args.bert_name, args.num_class, args.bert_type, args.drop_p)
    
    return model

def preprocess(logger):
    if args.bert_type != 'no_bert':
        task2dataset = {'SNLI': SNLIDataset, 'Pretrain': BertDataset}
        Dataset = task2dataset.get(args.task)
        Dataset = Dataset if Dataset != None else GlueDataset

        # get dataset
        if(Dataset in [SNLIDataset, GlueDataset]):
            train_dataset = Dataset(args.data_dir, args.task, args.max_len, args.bert_name, args.bert_type, mode='train')
            dev_dataset = Dataset(args.data_dir, args.task, args.max_len, args.bert_name, args.bert_type, mode='test')
        elif(Dataset == BertDataset):
            train_dataset = Dataset(os.path.join(args.data_dir, 'train.h5'))
            dev_dataset = Dataset(os.path.join(args.data_dir, 'dev.h5'))
        else:
            print('no selected dataset.')
    else:
        train_dataset = NormalDataset(args.task, args.max_len, mode='train')
        dev_dataset = NormalDataset(args.task, args.max_len, mode='validation', word2idx=train_dataset.word2idx)

    # save dataset
    save_path = os.path.join(args.data_dir, args.task, '_'.join([args.bert_type, args.bert_name]))
    torch.save(train_dataset, save_path + '_train.pt')
    torch.save(dev_dataset, save_path + '_dev.pt')

def train(logger):
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(train_job, nprocs=args.ngpus_per_node, args=(args, logger, ))
    else:
        train_job(gpu=None, args=args, logger=logger)

def train_job(gpu, args, logger):
    # log some information
    logger.info('-' * 40)
    logger.info((' '.join(sys.argv[1:])))
    logger.info('-' * 40)

    # set distribute training environment
    args.gpu = gpu
    if args.multiprocessing_distributed:
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.local_rank)

    # prepare dataset and dataloader
    data_path = os.path.join(args.data_dir, args.task, '_'.join([args.bert_type, args.bert_name]))
    train_dataset = torch.load(data_path + '_train.pt')
    dev_dataset = torch.load(data_path + '_dev.pt')

    if args.clip_dataset:
        train_size, dev_size = int(len(train_dataset) * 0.1), int(len(dev_dataset) * 0.3)
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
        dev_dataset, _ = torch.utils.data.random_split(dev_dataset, [dev_size, len(dev_dataset) - dev_size])

    try:
        num_class = train_dataset.num_class
    except:
        pass

    try:
        n_vocab = len(train_dataset.word2idx)
    except:
        n_vocab = None

    # define model and optimzier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(args, n_vocab)
    if args.multiprocessing_distributed:
        model.cuda(args.gpu)
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = model.to(device)
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # resume training
    if args.resume:
        state = torch.load(args.saved_model)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])

    if args.adv_fix_embedding:
        params = [param for name, param in model.named_parameters() if 'Embedding' not in name]
        optimizer = torch.optim.Adam(params, lr=args.lr)

    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler == None), num_workers=args.workers, pin_memory=True, sampler=train_sampler), \
                                        DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    else:
        pass

    trainer = Trainer(train_dataloader, eval_dataloader, model, optimizer, logger, args)
    trainer.train()

if __name__ == '__main__':
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # define logger file
    logger = init_logger(args.log_path)

    if args.do_prepare:
        preprocess(logger)

    if args.do_train:
        train(logger)

    if not(args.do_train or args.do_prepare):
        print('Nothing have done!')
