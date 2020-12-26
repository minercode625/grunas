import torch
from torch.autograd import Variable
from utils import AverageMeter, accuracy
import time


class TrainerSupernet:
    def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, train_param):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.param = AverageMeter()
        self.losses = AverageMeter()
        self.losses_lat = AverageMeter()
        self.losses_ce = AverageMeter()

        self.criterion = criterion
        self.w_optimizer = w_optimizer
        self.theta_optimizer = theta_optimizer
        self.w_scheduler = w_scheduler

        self.temperature = train_param['temperature']
        self.exp_anneal_rate = train_param['exp_anneal_rate']  # apply it every epoch
        self.cnt_epochs = train_param['cnt_epochs']
        self.train_thetas_from_the_epoch = train_param['train_thetas_from_the_epoch']
        self.path_to_save_model = train_param['path_to_save_model']

    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):
        
        best_top1 = 0.0
        # firstly, train weights only
        start_time = time.time()

        for epoch in range(self.train_thetas_from_the_epoch):
            loss_val, ce_val, param_val, top1_avg, _ = self._training_step(model, train_w_loader, self.w_optimizer)
            print('Epoch[{cur_epoch}/{max_epoch}] Pretrain Train: Loss: {loss:.4f}, CE: {ce:.4f}, param: {param:.4f}, acc: {top1:.4f}'.format(
                cur_epoch=epoch,
                max_epoch=self.cnt_epochs,
                loss=loss_val,
                ce=ce_val,
                param=param_val,
                top1=top1_avg                
            ))
            self.w_scheduler.step()
        print("pretrain finished")

        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):

            loss_val, ce_val, param_val, top1_avg, _ = self._training_step(model, train_w_loader, self.w_optimizer)
            self.w_scheduler.step()
            print('Epoch[{cur_epoch}/{max_epoch}] Weight Train: Loss: {loss:.4f}, CE: {ce:.4f}, param: {param:.4f}, acc: {top1:.4f}'.format(
                cur_epoch=epoch,
                max_epoch=self.cnt_epochs,
                loss=loss_val,
                ce=ce_val,
                param=param_val,
                top1=top1_avg                
            ))
            loss_val, ce_val, param_val, top1_avg, param_avg = self._training_step(model, train_thetas_loader, self.theta_optimizer)
            print('Epoch[{cur_epoch}/{max_epoch}] Thetas Train: Loss: {loss:.4f}, CE: {ce:.4f}, param: {param:.4f}, acc: {top1:.4f}'.format(
                cur_epoch=epoch,
                max_epoch=self.cnt_epochs,
                loss=loss_val,
                ce=ce_val,
                param=param_val,
                top1=top1_avg                
            ))
            top1_avg = self._validate(model, test_loader)
            print('top_avg: {top1_avg:.4f} param_avg: {param:.4f}'.format(
                top1_avg=top1_avg,
                param=param_avg
            ))
            if best_top1 < top1_avg:
                best_top1 = top1_avg
                torch.save(model.state_dict(), self.path_to_save_model)
            self.temperature = self.temperature * self.exp_anneal_rate
            print('Group Info: ' + model.get_max_group())
        print('Time:',time.time() - start_time)
        print('Best Top1: ' + str(best_top1))

    def _training_step(self, model, loader, optimizer):
        model = model.train()
        loss_list = AverageMeter()
        ce_list = AverageMeter()
        param_list = AverageMeter()
        for step, (X, y) in enumerate(loader):

            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            N = X.shape[0]

            optimizer.zero_grad()
            outs, parameters_to_accumulate = model(X, self.temperature)
            parameters_to_accumulate = torch.mean(parameters_to_accumulate)
            self.param.update(parameters_to_accumulate.item(), N)
            loss, ce, param = self.criterion(outs, y, parameters_to_accumulate)
            loss_list.update(loss.item(), X.size(0))
            ce_list.update(ce.item(), X.size(0))
            param_list.update(param.item(), X.size(0))
            loss.backward()
            optimizer.step()

            self._intermediate_stats_logging(outs, y, loss, N)
        top1_avg = self.top1.get_avg()
        param_avg = self.param.get_avg()
        for avg in [self.top1, self.top5, self.losses, self.param]:
            avg.reset()
        return loss_list.get_avg(), ce_list.get_avg(), param_list.get_avg(), top1_avg, param_avg

    def _validate(self, model, loader):
        model.eval()
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.cuda(), y.cuda()
                N = X.shape[0]

                outs, parameters_to_accumulate = model(X, self.temperature)
                parameters_to_accumulate = torch.mean(parameters_to_accumulate)
                loss, _, _ = self.criterion(outs, y, parameters_to_accumulate)

                self._intermediate_stats_logging_test(outs, y, loss, N)

        top1_avg = self.top1.get_avg()

        for avg in [self.top1, self.top5, self.losses]:
            avg.reset()
        return top1_avg

    def _intermediate_stats_logging(self, outs, y, loss, N):
        prec1, prec3 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top5.update(prec3.item(), N)

    def _intermediate_stats_logging_test(self, outs, y, loss, N):
        prec1, prec3 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top5.update(prec3.item(), N)


