import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, Criterion
from sklearn.linear_model import LogisticRegression


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        
        self.lambda_e = config['trainer']['lambda_e']
        self.lambda_od = config['trainer']['lambda_od']
        self.gamma_e = config['trainer']['gamma_e']
        self.gamma_od = config['trainer']['gamma_od']
        self.step_size = config['trainer']['step_size']
        self.dataset_name = config['data_loader']['type']
        self.living_classes = [2,3,4,5,6,7]

        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.criterion = Criterion(self.lambda_e, self.lambda_od, self.gamma_e, self.gamma_od, self.step_size)
        self.target_clf = LogisticRegression()
        self.sensitive_clf = LogisticRegression()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        if self.dataset_name == 'CIFAR10DataLoader':
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                sensitive = torch.tensor([i in self.living_classes for i in target]).long()

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target, sensitive, self.dataset_name, batch_idx)
                loss.backward()
                self.optimizer.step()
            
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))

                if batch_idx == self.len_epoch:
                    break
        else:
            for batch_idx, (data, sensitive, target) in enumerate(self.data_loader):
                data, sensitive, target = data.to(self.device), sensitive.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target, sensitive, self.dataset_name, batch_idx)
                loss.backward()
                self.optimizer.step()
            
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))

                if batch_idx == self.len_epoch:
                    break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if self.dataset_name == 'CIFAR10DataLoader':
                for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    sensitive = torch.tensor([i in self.living_classes for i in target]).long()
    
                    output = self.model(data)
                    loss = self.criterion(output, target, sensitive, self.dataset_name, batch_idx)

                    z_t = output[2][0]
                    t_clf = self.target_clf.fit(z_t, target)
                    t_predictions = torch.tensor(t_clf.predict(z_t))
                    s_clf = self.sensitive_clf.fit(z_t, sensitive)
                    s_predictions = torch.tensor(s_clf.predict(z_t))

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.valid_metrics.update('loss', loss.item())
                    self.valid_metrics.update('accuracy', self.metric_ftns[0](t_predictions, target))
                    self.valid_metrics.update('sens_accuracy', self.metric_ftns[1](s_predictions, sensitive))

            else:
                for batch_idx, (data, sensitive, target) in enumerate(self.valid_data_loader):
                    data, sensitive, target = data.to(self.device), sensitive.to(self.device), target.to(self.device)
                    #import pdb; pdb.set_trace()
                    output = self.model(data)
                    loss = self.criterion(output, target, sensitive, self.dataset_name, batch_idx)

                    z_t = output[2][0]
                    t_clf = self.target_clf.fit(z_t, target)
                    t_predictions = torch.tensor(t_clf.predict(z_t))
                    s = torch.argmax(sensitive, dim=1)
                    s_clf = self.sensitive_clf.fit(z_t, s)
                    s_predictions = torch.tensor(s_clf.predict(z_t))

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.valid_metrics.update('loss', loss.item())
                    self.valid_metrics.update('accuracy', self.metric_ftns[0](t_predictions, target))
                    self.valid_metrics.update('sens_accuracy', self.metric_ftns[1](s_predictions, s))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
