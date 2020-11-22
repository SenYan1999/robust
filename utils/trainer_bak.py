import torch
import torch.nn.functional as F
import numpy as np
import os
import time

from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, accuracy_score
from apex import amp
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, train_dataloader, dev_dataloader, model, pgd, pgd_k, smart, smart_alpha, optimizer, scheduler, task, logger, \
                 fp16, device, train_mode, tensorboard_path, load_path=None):
        self.train_data = train_dataloader
        self.dev_data = dev_dataloader
        self.model = model
        self.pgd = pgd
        self.pgd_k = pgd_k
        self.smart = smart
        self.smart_alpha = smart_alpha
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.task = task
        self.logger = logger
        self.train_mode = train_mode
        self.fp16 = fp16
        self.load_path = load_path
        self.writer = SummaryWriter(tensorboard_path)

        self.metrics = {'CoLA': 'MCC', 'QNLI': 'ACC', 'SST-2': 'ACC', 'MNLI': 'ACC', 'WNLI': 'ACC', 'QQP': 'ACC',\
                        'MRPC': 'ACC', 'RTE': 'ACC', 'SNLI': 'ACC'}

    def calculate_result(self, pred, truth):
        pred = torch.argmax(pred, dim=-1).detach().cpu().numpy().astype(np.float)
        truth = truth.detach().cpu().numpy().astype(np.float)

        if(self.task == 'Pretrain'):
            idx = np.where(truth != -1)
            pred = pred[idx]
            truth = truth[idx]

        if self.task == 'CoLA':
            result = matthews_corrcoef(truth, pred)
        elif self.task in ['QNLI', 'SST-2', 'WNLI', 'QQP', 'MNLI', 'MRPC', 'RTE', 'SNLI', 'Pretrain']:
            result = accuracy_score(truth, pred)
        else:
            raise('Task error!')

        return result

    def train_epoch_pgd(self, epoch, print_interval):
        self.logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data), position=0, leave=True)
        self.model.train()

        losses, accs = [], []
        t0 = time.time()
        for step, batch in enumerate(self.train_data):
            input_ids, attention_mask, token_type_ids, labels = map(lambda i: i.to(self.device), batch)

            out = self.model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(out, labels)

            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)

            # PGD Adversarial Training
            self.pgd.grad_backup()
            for k in range(self.pgd_k):
                self.pgd.attack(is_first_attack=(k == 0))
                if k != self.pgd_k - 1:
                    self.model.zero_grad()
                else:
                    self.pgd.restore_grad()
                out_adv = self.model(input_ids, attention_mask, token_type_ids)
                loss_adv = F.cross_entropy(out_adv, labels)

                if self.fp16:
                    with amp.scale_loss(loss_adv, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_adv.backward()

            self.pgd.restore()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())
            acc = self.calculate_result(out, labels)
            accs.append(acc)

            t1 = time.time()
            pbar.set_description('Epoch: %2d | LOSS: %2.3f | %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))
            pbar.update(1)
            if step % print_interval == 0 and step != 0:
                total_time = (t1 - t0) / (step / len(self.train_data))
                self.logger.info('Epoch: %2d | Step: [%4d / %4d] | Time: [%5.3f s / % 5.3f s] | LOSS: %2.3f | %s: %1.3f' % \
                                 (epoch, step, len(self.train_data), (t1 - t0), total_time, np.mean(losses), self.metrics[self.task], np.mean(accs)))

        pbar.close()
        self.logger.info('Epoch: %2d | LOSS: %2.3f %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))

    def train_epoch_smart(self, epoch, print_interval):
        self.logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data), position=0, leave=True)
        self.model.train()

        losses, accs = [], []
        for batch in self.train_data:
            input_ids, attention_mask, token_type_ids, labels = map(lambda i: i.to(self.device), batch)

            out = self.model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(out, labels)

            # smart loss
            adv_loss = self.smart.forward(self.model, out, input_ids, token_type_ids, attention_mask)
            loss += self.smart_alpha * adv_loss

            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())
            acc = self.calculate_result(out, labels)
            accs.append(acc)

            pbar.set_description('Epoch: %2d | LOSS: %2.3f | %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))
            pbar.update(1)

        pbar.close()
        self.logger.info('Epoch: %2d | LOSS: %2.3f %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))

    def train_epoch_normal(self, epoch, print_interval):
        self.logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data), position=0, leave=True)
        self.model.train()

        losses, accs = [], []
        for batch in self.train_data:
            input_ids, attention_mask, token_type_ids, labels = map(lambda i: i.to(self.device), batch)

            out = self.model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(out, labels)
            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())
            acc = self.calculate_result(out, labels)
            accs.append(acc)

            pbar.set_description('Epoch: %2d | LOSS: %2.3f | %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))
            pbar.update(1)

        pbar.close()
        self.logger.info('Epoch: %2d | LOSS: %2.3f %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))

    def evaluate_epoch(self, epoch):
        self.logger.info('Epoch: %2d: Evaluating Model...' % epoch)
        self.model.eval()

        losses, precise, recall, f1s, accs = [], [], [], [], []
        for batch in self.dev_data:
            input_ids, attention_mask, token_type_ids, labels = map(lambda i: i.to(self.device), batch)

            with torch.no_grad():
                out = self.model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(out, labels)

            acc = self.calculate_result(out, labels)
            losses.append(loss.item())
            accs.append(acc)

        self.logger.info('Epoch: %2d | LOSS: %2.3f %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))

    def train(self, num_epoch, save_path, print_interval):
        for epoch in range(num_epoch):
            if self.train_mode == 'normal':
                self.train_epoch_normal(epoch, print_interval)
            elif self.train_mode in ['pgd', 'fix']:
                self.train_epoch_pgd(epoch, print_interval)
            elif self.train_mode == 'smart':
                self.train_epoch_smart(epoch, print_interval)
            else:
                assert Exception('Please at least one training method.')

            self.evaluate_epoch(epoch)

            # save state dict
            path = os.path.join(save_path, 'state_%d_epoch.pt' % epoch)
            self.save_dict(path)
            self.logger.info('')

    def save_dict(self, save_path):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state_dict, save_path)

    def load_dict(self, path, optimizer=None):
        print('load dict...')
        state_dict = torch.load(path)

        self.model.load_state_dict(state_dict['model'])
        if optimizer == None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
            self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            pass

class OnlyAdversarialTrainer(Trainer):
    def __init__(self, train_dataloader, dev_dataloader, model, pgd, pgd_k, optimizer, scheduler, task, logger, \
                fp16, device, train_mode, tensorboard_path, load_path=None):
        super(OnlyAdversarialTrainer, self).__init__(train_dataloader, dev_dataloader, model, pgd, pgd_k, optimizer, scheduler, task, logger, \
                fp16, device, train_mode, tensorboard_path, load_path)
        self.load_dict(load_path, optimizer)
    
    def train_pgd(self, epoch, print_interval):
        self.logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data), position=0, leave=True)
        self.model.train()

        losses, accs = [], []
        t0 = time.time()
        for step, batch in enumerate(self.train_data):
            input_ids, attention_mask, token_type_ids, labels = map(lambda i: i.to(self.device), batch)

            out = self.model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(out, labels)

            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # PGD Adversarial Training
            self.pgd.grad_backup()
            for k in range(self.pgd_k):
                self.pgd.attack(is_first_attack=(k == 0))
                if k != self.pgd_k - 1:
                    self.model.zero_grad()

                out_adv = self.model(input_ids, attention_mask, token_type_ids)
                loss_adv = F.cross_entropy(out_adv, labels)

                if self.fp16:
                    with amp.scale_loss(loss_adv, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_adv.backward()

            self.pgd.restore()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())
            acc = self.calculate_result(out, labels)
            accs.append(acc)

            t1 = time.time()
            pbar.set_description('Epoch: %2d | LOSS: %2.3f | %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))
            pbar.update(1)
            if step % print_interval == 0 and step != 0:
                total_time = (t1 - t0) / (step / len(self.train_data))
                self.logger.info('Epoch: %2d | Step: [%4d / %4d] | Time: [%5.3f s / % 5.3f s] | LOSS: %2.3f | %s: %1.3f' % \
                                 (epoch, step, len(self.train_data), (t1 - t0), total_time, np.mean(losses), self.metrics[self.task], np.mean(accs)))

        pbar.close()
        self.logger.info('Epoch: %2d | LOSS: %2.3f %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))

class PretrainTrainer(Trainer):
    def __init__(self, train_dataloader, dev_dataloader, model, pgd, pgd_k, smart, smart_alpha, optimizer, scheduler, task, logger, \
                 fp16, device, train_mode, tensorboard_path, load_path=None):
        super(PretrainTrainer, self).__init__(train_dataloader, dev_dataloader, model, pgd, pgd_k, smart, smart_alpha, optimizer, scheduler, task, logger, \
                 fp16, device, train_mode, tensorboard_path, load_path=None)
    
    def train_epoch_pgd(self, epoch, print_interval):
        self.logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data), position=0, leave=True)
        self.model.train()

        losses, accs = {'mlm': [], 'nsp': []}, {'mlm': [], 'nsp': []}
        t0 = time.time()
        for step, batch in enumerate(self.train_data):
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels = \
                map(lambda x: x.to(self.device), batch)
            labels = {'mlm': masked_lm_labels, 'nsp': next_sentence_labels}

            for task in ['mlm']:
                out = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.model.get_loss(out, labels, task)

                if self.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # PGD Adversarial Training
                self.pgd.grad_backup()
                for k in range(self.pgd_k):
                    self.pgd.attack(is_first_attack=(k == 0))
                    if k != self.pgd_k - 1:
                        self.model.zero_grad()
                    else:
                        self.pgd.restore_grad()
                    out_adv = self.model(input_ids, attention_mask, token_type_ids)
                    loss_adv = self.model.get_loss(out_adv, labels, task)

                    if self.fp16:
                        with amp.scale_loss(loss_adv, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_adv.backward()

                self.pgd.restore()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                losses[task].append(loss.item())
                acc = self.calculate_result(out[task].reshape(-1, out[task].shape[-1]), labels[task].reshape(-1))
                accs[task].append(acc)

                # tensorboard
                self.writer.add_scalar(task + ': loss', np.mean(losses[task]), len(self.train_data) * (epoch + 1) + step)
                self.writer.add_scalar(task + ': acc', np.mean(accs[task]), len(self.train_data) * (epoch + 1) + step)

            t1 = time.time()

            pbar.set_description('Epoch: %2d | MLM LOSS: %2.3f | MLM ACC: %1.3f NSP LOSS: %2.3f | NSP ACC: %1.3f' \
                % (epoch, np.mean(losses['mlm']), np.mean(accs['mlm']), np.mean(losses['nsp']), np.mean(accs['nsp'])))
            pbar.update(1)

            if step % print_interval == 0 and step != 0:
                total_time = (t1 - t0) / (step / len(self.train_data))
                self.logger.info('Epoch: %2d | Step: [%4d / %4d] | Time: [%5.3f s / % 5.3f s] | MLM LOSS: %2.3f | \
                    MLM ACC: %1.3f NSP LOSS: %2.3f | NSP ACC: %1.3f' % (epoch, step, len(self.train_data), (t1 - t0), \
                        total_time, np.mean(losses['mlm']), np.mean(accs['mlm']), np.mean(losses['nsp']), np.mean(accs['nsp'])))

        pbar.close()
        self.logger.info('Epoch: %2d | MLM LOSS: %2.3f | MLM ACC: %1.3f NSP LOSS: %2.3f | NSP ACC: %1.3f' % \
            (epoch, np.mean(losses['mlm']), np.mean(accs['mlm']), np.mean(losses['nsp']), np.mean(accs['nsp'])))

    def train_epoch_smart(self, epoch, print_interval):
        self.logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data), position=0, leave=True)
        self.model.train()
        t0 = time.time()

        losses, accs = {'mlm': [], 'nsp': []}, {'mlm': [], 'nsp': []}
        for step, batch in enumerate(self.train_data):
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels = \
                map(lambda x: x.to(self.device), batch)
            labels = {'mlm': masked_lm_labels, 'nsp': next_sentence_labels}

            for task in ['mlm', 'nsp']:
                out = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.model.get_loss(out, labels, task)

                # smart loss
                adv_loss = self.smart.forward(self.model, out, input_ids, token_type_ids, attention_mask)
                loss += self.smart_alpha * adv_loss

                if self.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                losses[task].append(loss.item())
                acc = self.calculate_result(out[task].reshape(-1, out[task].shape[-1]), labels[task].reshape(-1))
                accs[task].append(acc)

                # tensorboard
                self.writer.add_scalar(task + ': loss', np.mean(losses[task]), len(self.train_data) * (epoch + 1) + step)
                self.writer.add_scalar(task + ': acc', np.mean(accs[task]), len(self.train_data) * (epoch + 1) + step)

            t1 = time.time()

            pbar.set_description('Epoch: %2d | MLM LOSS: %2.3f | MLM ACC: %1.3f NSP LOSS: %2.3f | NSP ACC: %1.3f' \
                % (epoch, np.mean(losses['mlm']), np.mean(accs['mlm']), np.mean(losses['nsp']), np.mean(accs['nsp'])))
            pbar.update(1)

            if step % print_interval == 0 and step != 0:
                total_time = (t1 - t0) / (step / len(self.train_data))
                self.logger.info('Epoch: %2d | Step: [%4d / %4d] | Time: [%5.3f s / % 5.3f s] | MLM LOSS: %2.3f | \
                    MLM ACC: %1.3f NSP LOSS: %2.3f | NSP ACC: %1.3f' % (epoch, step, len(self.train_data), (t1 - t0), \
                        total_time, np.mean(losses['mlm']), np.mean(accs['mlm']), np.mean(losses['nsp']), np.mean(accs['nsp'])))

    def train_epoch_normal(self, epoch, print_interval):
        self.logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data), position=0, leave=True)
        self.model.train()

        t0 = time.time()

        losses, accs = {'mlm': [], 'nsp': []}, {'mlm': [], 'nsp': []}
        for step, batch in enumerate(self.train_data):
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels = \
                map(lambda x: x.to(self.device), batch)
            labels = {'mlm': masked_lm_labels, 'nsp': next_sentence_labels}

            for task in ['mlm']:
                out = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.model.get_loss(out, labels, task)

                if self.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                losses[task].append(loss.item())
                acc = self.calculate_result(out[task].reshape(-1, out[task].shape[-1]), labels[task].reshape(-1))
                accs[task].append(acc)

                # tensorboard
                self.writer.add_scalar(task + ': loss', np.mean(losses[task]), len(self.train_data) * (epoch + 1) + step)
                self.writer.add_scalar(task + ': acc', np.mean(accs[task]), len(self.train_data) * (epoch + 1) + step)

            t1 = time.time()

            pbar.set_description('Epoch: %2d | MLM LOSS: %2.3f | MLM ACC: %1.3f NSP LOSS: %2.3f | NSP ACC: %1.3f' \
                % (epoch, np.mean(losses['mlm']), np.mean(accs['mlm']), np.mean(losses['nsp']), np.mean(accs['nsp'])))
            pbar.update(1)

            if step % print_interval == 0 and step != 0:
                total_time = (t1 - t0) / (step / len(self.train_data))
                self.logger.info('Epoch: %2d | Step: [%4d / %4d] | Time: [%5.3f s / % 5.3f s] | MLM LOSS: %2.3f | \
                    MLM ACC: %1.3f NSP LOSS: %2.3f | NSP ACC: %1.3f' % (epoch, step, len(self.train_data), (t1 - t0), \
                        total_time, np.mean(losses['mlm']), np.mean(accs['mlm']), np.mean(losses['nsp']), np.mean(accs['nsp'])))

    def evaluate_epoch(self, epoch):
        self.logger.info('Epoch: %2d: Evaluating Model...' % epoch)
        self.model.eval()

        losses, accs = {'mlm': [], 'nsp': []}, {'mlm': [], 'nsp': []}
        for batch in self.dev_data:
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels = \
                map(lambda x: x.to(self.device), batch)
            labels = {'mlm': masked_lm_labels, 'nsp': next_sentence_labels}

            with torch.no_grad():
                out = self.model(input_ids, attention_mask, token_type_ids)

            for task in ['mlm']:
                loss = self.model.get_loss(out, labels, task)

                acc = self.calculate_result(out[task].reshape(-1, out[task].shape[-1]), labels[task].reshape(-1))
                losses[task].append(loss.item())
                accs[task].append(acc)

        self.logger.info('Epoch: %2d | MLM LOSS: %2.3f | MLM ACC: %1.3f NSP LOSS: %2.3f | NSP ACC: %1.3f' % \
            (epoch, np.mean(losses['mlm']), np.mean(accs['mlm']), np.mean(losses['nsp']), np.mean(accs['nsp'])))

    def train(self, num_epoch, save_path, print_interval):
        for epoch in range(num_epoch):
            if self.train_mode == 'normal':
                self.train_epoch_normal(epoch, print_interval)
            elif self.train_mode in ['pgd', 'fix']:
                self.train_epoch_pgd(epoch, print_interval)
            elif self.train_mode == 'smart':
                self.train_epoch_smart(epoch, print_interval)
            else:
                assert Exception('Please at least one training method.')

            self.evaluate_epoch(epoch)

            # save state dict
            path = os.path.join(save_path, 'model.pt')
            self.save_dict(path)
            self.logger.info('')
