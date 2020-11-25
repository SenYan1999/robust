import torch
import os
import numpy as np
import torch.nn.functional as F

from model.perturbation import alum_perturbation, random_perturbation
from sklearn.metrics import matthews_corrcoef, accuracy_score
from tqdm import tqdm

class MeanTool:
    def __init__(self):
        self.metric = []
        self.batch_size = []

    def update(self, metric, batch_size):
        self.metric.append(metric)
        self.batch_size.append(batch_size)

    def compute_mean(self):
        return (np.sum(np.array(self.metric) * np.array(self.batch_size)) / np.sum(self.batch_size)).item()

def get_metric(args):
    if args.mlm_task:
        metric = accuracy_score
    elif args.task == 'cola':
        metric = matthews_corrcoef
    else:
        metric = accuracy_score
    return metric

def calculate_result(args, pred, truth):
    pred = torch.argmax(pred, dim=-1).reshape(-1)
    truth = truth.detach().reshape(-1)

    # remove pad
    if args.mlm_task:
        pos = torch.where(truth!=0)[0]
        pred = pred[pos].cpu().numpy().astype(np.float)
        truth = truth[pos].cpu().numpy().astype(np.float)

    metric = get_metric(args)
    result = metric(truth, pred)

    return result

class Trainer(object):
    def __init__(self, train_dataloader, dev_dataloader, model, optimizer, logger, args):
        self.train_data = train_dataloader
        self.dev_data = dev_dataloader
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.logger = logger

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(total=len(self.train_data))
        loss_mean_tool = MeanTool()
        metric_mean_tool = MeanTool()
        for step, batch in enumerate(self.train_data):
            # move data to cuda
            if self.args.gpu != None:
                batch = list(map(lambda x: x.cuda(), batch))
            else:
                batch = list(map(lambda x: x.cuda(self.args.gpu, non_blocking=True), batch))

            # apply model to the input batch
            logit, embed = self.model(batch)
            loss_normal = F.nll_loss(F.log_softmax(logit, dim=-1).reshape(-1, logit.shape[-1]), batch[-1].reshape(-1), ignore_index=-1)

            # now we apply adv perturbation
            if self.args.adv_train:
                loss_adv = alum_perturbation(self.model, batch, embed, logit, self.args)
                loss = loss_normal + loss_adv
            elif self.args.random_train:
                loss_random = random_perturbation(self.model, batch, embed, logit, self.args)
                loss = loss_normal + loss_random
            else:
                loss = loss_normal
            loss.backward()

            # optimize
            self.optimizer.step()

            # print log information
            metric_score = calculate_result(self.args, logit, batch[-1])
            metric_mean_tool.update(metric_score, batch[0].shape[0])
            loss_mean_tool.update(loss.item(), batch[0].shape[0])
            pbar.update(1)
            information = 'Epoch: %3d | Loss: %.3f | Metric: %.3f' % (epoch, loss.item(), metric_score)
            pbar.set_description(information)

            if step % 500 == 0:
                self.logger.info(information)

        pbar.close()
        self.logger.info('*' * 30)
        self.logger.info('Train Result:')
        self.logger.info('%10s: %3d' % ('Epoch', epoch))
        self.logger.info('%10s: %.3f' % ('Loss', loss_mean_tool.compute_mean()))
        self.logger.info('%10s: %.3f' % ('Metric', metric_mean_tool.compute_mean()))
        self.logger.info('*' * 30)

    def evaluate_epoch(self, epoch):
        self.model.eval()

        loss_mean_tool = MeanTool()
        metric_mean_tool = MeanTool()
        for batch in self.dev_data:
            if self.args.gpu != None:
                batch = list(map(lambda x: x.cuda(), batch))
            else:
                batch = list(map(lambda x: x.cuda(self.args.gpu, non_blocking=True), batch))

            with torch.no_grad():
                out, _ = self.model(batch)
            loss = F.nll_loss(F.log_softmax(out, dim=-1).reshape(-1, out.shape[-1]), batch[-1].reshape(-1), ignore_index=-1)

            metric_score = calculate_result(self.args, out, batch[-1])

            loss_mean_tool.update(loss.item(), batch_size=batch[0].shape[0])
            metric_mean_tool.update(metric_score, batch_size=batch[0].shape[0])

        self.logger.info('*' * 30)
        self.logger.info('Evaluate Result:')
        self.logger.info('%10s: %3d' % ('Epoch', epoch))
        self.logger.info('%10s: %.3f' % ('Loss', loss_mean_tool.compute_mean()))
        self.logger.info('%10s: %.3f' % ('Metric', metric_mean_tool.compute_mean()))
        self.logger.info('*' * 30)

        return metric_mean_tool.compute_mean()

    def train(self):
        best_metric = -100
        for epoch in range(self.args.num_epoch):
            self.train_epoch(epoch)
            metric_score = self.evaluate_epoch(epoch)

            if metric_score > best_metric:
                self.save_state()

            best_metric = max(best_metric, metric_score)
            self.logger.info('Best Metric: %.3f' % best_metric)
    
    def save_state(self):
        state = {'model_state': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict()}
        saved_model_name = type(self.model).__name__
        saved_model_name += '_adv' if self.args.adv_train else ''
        saved_model_name += '_fix_embed' if self.args.adv_fix_embedding else ''
        saved_model_name += '.pt'
        torch.save(state, os.path.join(self.args.save_dir, saved_model_name))
