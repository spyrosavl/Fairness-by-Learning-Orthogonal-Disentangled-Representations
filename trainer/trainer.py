import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, Criterion
from sklearn.linear_model import LogisticRegression
from model.model import *
from sklearn.preprocessing import normalize

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer_1, optimizer_2 , optimizer_3, optimizer_4, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer_1, optimizer_2, optimizer_3, optimizer_4, config)
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
        self.tar_clf = Cifar_Classifier(z_dim=128, hidden_dim=[256, 128], out_dim=2)
        self.sen_clf = Cifar_Classifier(z_dim=128, hidden_dim=[256, 128], out_dim=10)

        self.cross = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        if self.dataset_name == 'CIFAR10DataLoader':
            for batch_idx, (data, sensitive) in enumerate(self.data_loader):
                data, sensitive = data.to(self.device), sensitive.to(self.device)
                target = torch.tensor([i in self.living_classes for i in sensitive]).long()
                
                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
                output = self.model(data)
                s_zs = output[1][2]
                L_s = self.cross(s_zs, sensitive)
                
                for param in self.model.encoder.resnet.parameters():
                    param.requires_grad=False
                L_s.backward(retain_graph=True)

                for param in self.model.encoder.resnet.parameters():
                    param.requires_grad=True
                loss = self.criterion(output, target, sensitive, self.dataset_name, epoch)

                loss.backward()
                self.optimizer_1.step()
                self.optimizer_2.step()
            
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item()+L_s)

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()+L_s))

                if batch_idx == self.len_epoch:
                    break
        else:
            for batch_idx, (data, sensitive, target) in enumerate(self.data_loader):
                data, sensitive, target = data.to(self.device), sensitive.to(self.device), target.to(self.device)
                
                #import pdb; pdb.set_trace()
                self.optimizer_1.zero_grad()
                output = self.model(data)

                s_zs = output[1][2]
                L_s = self.bce(s_zs, sensitive.float())
                for param in self.model.encoder.shared_model.parameters():
                    param.requires_grad=False
                L_s.backward(retain_graph=True)

                for param in self.model.encoder.shared_model.parameters():
                    param.requires_grad=True
       
                loss = self.criterion(output, target, sensitive, self.dataset_name, batch_idx)
                loss.backward()
                self.optimizer_1.step()
                
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item()+L_s)

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
        #with torch.no_grad():
        if self.dataset_name == 'CIFAR10DataLoader':
            for batch_idx, (data, sensitive) in enumerate(self.valid_data_loader):
                data, sensitive = data.to(self.device), sensitive.to(self.device)
                target = torch.tensor([i in self.living_classes for i in sensitive]).long()
                
                self.optimizer_3.zero_grad()
                self.optimizer_4.zero_grad()

                output = self.model(data)
                z_t = output[2][0]
                
                t_predictions = self.tar_clf.forward(z_t)
                t_pred = torch.argmax(t_predictions, dim=1)
                loss_clf_1 = self.cross(t_predictions, target)
                loss_clf_1.backward(retain_graph=True)
                self.optimizer_3.step()

                s_predictions = self.sen_clf.forward(z_t)
                s_pred = torch.argmax(s_predictions, dim=1)
                loss_clf_2 = self.cross(s_predictions, sensitive)
                loss_clf_2.backward()
                self.optimizer_4.step()
                
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('accuracy', self.metric_ftns[0](t_pred, target))
                self.valid_metrics.update('sens_accuracy', self.metric_ftns[1](s_pred, sensitive))

        else:

            with torch.no_grad():
                for batch_idx, (data, sensitive, target) in enumerate(self.valid_data_loader):
                    data, sensitive, target = data.to(self.device), sensitive.to(self.device), target.to(self.device)
                    output = self.model(data)

                    s_zt = output[1][1]
                    L_s = self.bce(s_zt, sensitive.float())

                    loss = self.criterion(output, target, sensitive, self.dataset_name, batch_idx)

                    #import pdb; pdb.set_trace()
                    z_t = output[2][0]

                    t_clf = self.target_clf.fit(z_t, target)
                    t_predictions = torch.tensor(t_clf.predict(z_t))
                    s = torch.argmax(sensitive, dim=1)
                    s_clf = self.sensitive_clf.fit(z_t, s)
                    s_predictions = torch.tensor(s_clf.predict(z_t))

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.valid_metrics.update('loss', loss.item() + L_s)
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
