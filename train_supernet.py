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
            self._training_step(model, train_w_loader, self.w_optimizer)
            self.w_scheduler.step()
        print("pretrain finished")

        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):

            self._training_step(model, train_w_loader, self.w_optimizer)
            self.w_scheduler.step()

            self._training_step(model, train_thetas_loader, self.theta_optimizer)

            top1_avg = self._validate(model, test_loader)
            print('top_avg: ' + str(top1_avg) + ' param_avg: ' + str(self.param.get_avg()))
            if best_top1 < top1_avg:
                best_top1 = top1_avg
                torch.save(model.state_dict(), self.path_to_save_model)
            self.temperature = self.temperature * self.exp_anneal_rate
        
        print('Time:',time.time() - start_time)
        print('Best Top1: ' + str(best_top1))

    def _training_step(self, model, loader, optimizer):
        model = model.train()

        for step, (X, y) in enumerate(loader):

            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            N = X.shape[0]

            optimizer.zero_grad()
            parameters_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
            outs, parameters_to_accumulate = model(X, self.temperature, parameters_to_accumulate)
            self.param.update(parameters_to_accumulate.item(), N)
            loss = self.criterion(outs, y, parameters_to_accumulate)
            loss.backward()
            optimizer.step()

            self._intermediate_stats_logging(outs, y, loss, N)

        for avg in [self.top1, self.top5, self.losses]:
            avg.reset()

    def _validate(self, model, loader):
        model.eval()
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.cuda(), y.cuda()
                N = X.shape[0]

                latency_to_accumulate = torch.Tensor([[0.0]]).cuda()
                outs, latency_to_accumulate = model(X, self.temperature, latency_to_accumulate)
                loss = self.criterion(outs, y, latency_to_accumulate)

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


