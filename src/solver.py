import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from vadam.optimizers import Vadam

from src.model.build_gen import PhiGnet, QWnet
from src.data_manager.dataset_read import dataset_read
from src.utils.constants import device


###############################################################################

class Solver(object):

    ########
    def __init__(self, args, batch_size=128, source='svhn', target='mnist',
                 nsamps_q=50, lamb_marg_loss=10.0,
                 learning_rate=0.0002, interval=100, optimizer='adam', num_k=4, num_kq=4,
                 all_use=False, checkpoint_dir=None, save_epoch=10, use_vadam=False):

        # set hyperparameters
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.num_kq = num_kq
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.use_vadam = use_vadam
        if self.source == 'svhn':
            self.scale = True
        else:
            self.scale = False
        self.lamb_marg_loss = lamb_marg_loss

        # load data, do image transform, and create a mini-batch generator 
        print('dataset loading')
        self.datasets, self.dataset_test = dataset_read(source, target, self.batch_size,
                                                        scale=self.scale, all_use=self.all_use)
        print('load finished!')

        if source == 'svhn':
            self.Ns = 73257

        # create models
        self.phig = PhiGnet(source=source, target=target)
        self.qw = QWnet(source=source, target=target)

        # load the previously learned models from files (if evaluations only)
        if args.eval_only:
            self.phig = torch.load('%s/model_epoch%s_phig.pt' %
                                   (self.checkpoint_dir, args.resume_epoch))
            self.qw = torch.load('%s/model_epoch%s_qw.pt' %
                                 (self.checkpoint_dir, args.resume_epoch))

        # move models to GPU
        self.phig.to(device)
        self.qw.to(device)

        # create optimizer objects (one for each model)
        self.set_optimizer(which_opt=optimizer, lr=learning_rate)

        # print stats every interval (default: 100) minibatch iters
        self.interval = interval

        self.lr = learning_rate

        # some dimensions
        self.p = self.phig.p  # dim(phi(G(x)))
        self.M = nsamps_q  # number of samples from variational density q(w)

    ########
    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):

        if which_opt == 'momentum':
            self.opt_phig = optim.SGD(self.phig.parameters(),
                                      lr=lr, weight_decay=0., momentum=momentum)

            self.opt_qw = optim.SGD(self.qw.parameters(),
                                    lr=lr, weight_decay=0., momentum=momentum)

        if which_opt == 'adam':
            self.opt_phig = optim.Adam(self.phig.parameters(), lr=lr, weight_decay=0.)
            # self.opt_qw = optim.Adam(self.qw.parameters(), lr=lr, weight_decay=0.)
            # self.opt_phig = optim.Adam(self.phig.parameters(), lr=lr, weight_decay=0., amsgrad=True)
            if self.use_vadam:
                self.opt_qw = Vadam(self.qw.parameters(), 500*128, lr=0.0002)
            else:
                self.opt_qw = optim.Adam(self.qw.parameters(), lr=lr, weight_decay=0., amsgrad=True)

    ########
    def reset_grad(self):

        # zero out all gradients of model params registered in the optimizers
        self.opt_phig.zero_grad()
        self.opt_qw.zero_grad()

    ########
    def ent(self, output):

        return -torch.mean(output * torch.log(output + 1e-6))

    ########
    def kl_loss(self):

        kl = 0.5 * (-self.p * 10 +
                    torch.sum((torch.exp(self.qw.logsd)) ** 2 + self.qw.mu ** 2 - 2.0 * self.qw.logsd))

        return kl

    ########
    def train(self, epoch, record_file=None):
        """
        train models for one epoch (ie, one pass of whole training data)
        """
        criterion = nn.CrossEntropyLoss().to(device)
        correct = 0
        n_sample = 0
        self.phig.train()
        self.qw.train()

        torch.cuda.manual_seed(1)
        loss_traj = []
        # for each batch
        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            label_t = data["T_label"]
            if img_s.size()[0] < self.batch_size or \
                    img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.to(device)
            img_t = img_t.to(device)
            label_s = label_s.long().to(device)
            # imgs = Variable(torch.cat((img_s, img_t), 0))
            # label_s = Variable(label_s.long().to(device))
            # img_s = Variable(img_s)
            # img_t = Variable(img_t)

            # (M x p x K) samples from N(0,1)
            # eps = Variable(torch.randn(self.M, self.p, 10))
            eps = torch.randn(self.M, self.p, 10)
            eps = eps.to(device)

            # ### step A: min_{qw} (nll + kl) ###
            self.reset_grad()
            for i in range(self.num_kq):
                phig_s = self.phig(img_s)  # phi(G(xs))
                wsamp = self.qw(eps)  # samples from q(w)

                if self.use_vadam:
                    def closure():
                        wphig_s = torch.sum(wsamp.unsqueeze(1) * phig_s.unsqueeze(0).unsqueeze(3), dim=2)
                        loss = criterion(wphig_s.view(-1, 10), label_s.repeat(self.M)) * self.Ns
                        loss.backward()
                        return loss

                    # update models
                    loss_nll = self.opt_qw.step(closure)
                else:
                # w'*phi(G(xs)) = (M x B x K)
                    wphig_s = torch.sum(wsamp.unsqueeze(1) * phig_s.unsqueeze(0).unsqueeze(3), dim=2)

                # nll loss
                    loss_nll = criterion(wphig_s.view(-1, 10), label_s.repeat(self.M)) * self.Ns
                # loss_nll = criterion(wphig_s.view(-1, 10), label_s.repeat(self.M))

                # kl loss
                    loss_kl = self.kl_loss()

                    loss = loss_nll + loss_kl

                # compute gradient of the loss
                    loss.backward()
                    self.opt_qw.step()

                self.reset_grad()

            if batch_idx % 10 == 0:
                aloss = loss_nll.detach().clone().cpu().numpy()
                loss_traj.append(aloss)
            # ### step B: min_{phig} (nll + kl + marg) ###
            self.reset_grad()
            for i in range(self.num_k):
                phig_s = self.phig(img_s)  # phi(G(xs))
                phig_t = self.phig(img_t)  # phi(G(xt))
                wsamp = self.qw(eps)  # samples from q(w)

                # w'*phi(G(xs)) = (M x B x K)
                wphig_s = torch.sum(wsamp.unsqueeze(1) * phig_s.unsqueeze(0).unsqueeze(3), dim=2)

                # nll loss
                loss_nll = criterion(wphig_s.view(-1, 10), label_s.repeat(self.M)) * self.Ns
                # loss_nll = criterion(wphig_s.view(-1, 10), label_s.repeat(self.M))

                # kl loss
                loss_kl = self.kl_loss()

                # margin loss on target
                f_t = torch.mm(phig_t, self.qw.mu)  # (B x K)
                top2 = torch.topk(f_t, k=2, dim=1)[0]  # (B x 2)
                # top2[i,0] = max_j f_t[i,j], top2[:,1] = max2_j f_t[i,j]
                gap21 = top2[:, 1] - top2[:, 0]  # B-dim
                std_f_t = torch.sqrt(torch.mm(phig_t ** 2, torch.exp(self.qw.logsd) ** 2))  # (B x K)
                max_std = torch.max(std_f_t, dim=1)[0]  # B-dim
                loss_marg = torch.mean(F.relu(1.0 + gap21 + 1.96 * max_std))

                loss = loss_nll + loss_kl + self.lamb_marg_loss * loss_marg

                # compute gradient of the loss
                loss.backward()

                # update models
                self.opt_phig.step()
                self.reset_grad()

            with torch.no_grad():
                phig_s = self.phig(img_s)
                f_t = torch.mm(phig_s, self.qw.mu)
                pred = torch.argmax(f_t, dim=1)
                correct += pred.eq(label_s).sum().cpu().numpy()
                n_sample += pred.size(0)

            if batch_idx > 500:
                # record for the epoch
                vadam_str = "vadam" if self.use_vadam else "adam"
                loss_str = f"{vadam_str}_loss.txt"
                str_arr = ",".join(map(str, loss_traj))
                record = open(f"record/{loss_str}", 'a')
                record.write('%s\n' % (str_arr,))
                return batch_idx

            if batch_idx % self.interval == 0:
                prn_str = f"Train Epoch: {epoch} [batch-idx: {batch_idx}] " \
                          f"nll: {loss_nll.item(): .6f},  kl: {loss_kl.item(): .6f},  marg: {loss_marg.item(): .6f}, " \
                          f"S_accuracy: {100 * correct / n_sample: .4f}%"
                print(prn_str)
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s\n' % (prn_str,))
                    record.close()

        vadam_str = "vadam" if self.use_vadam else "adam"
        loss_str = f"{vadam_str}_loss.txt"
        str_arr = ",".join(map(str, loss_traj))
        record = open(f"record/{loss_str}", 'a')
        record.write('%s\n' % (str_arr,))
        return batch_idx

    ########
    def test(self, epoch, record_file=None, save_model=False):

        """
        evaluate the current models on the entire test set
        """

        criterion = nn.CrossEntropyLoss().to(device)

        # turn models into evaluation mode
        self.phig.eval()
        self.qw.eval()

        test_loss = 0  # test nll loss
        corrects = 0  # number of correct predictions by MAP
        size = 0  # total number of test samples

        s_corrects = 0
        s_size = 0
        # turn off autograd feature (no evaluation history tracking)
        with torch.no_grad():

            for batch_idx, data in enumerate(self.dataset_test):
                img = data['T']
                label = data['T_label']
                img_s = data["S"]
                label_s = data["S_label"]

                img, label = img.to(device), label.long().to(device)
                img_s = img_s.to(device)
                label_s = label_s.long().to(device)
                # img, label = Variable(img, volatile=True), Variable(label)
                # img, label = Variable(img), Variable(label)

                # (M x p x K) samples from N(0,1)
                # eps = Variable(torch.randn(self.M, self.p, 10))
                # eps = eps.cuda()
                phig = self.phig(img)  # phi(G(x))
                wmode = self.qw.mu  # mode of q(w)
                # wsamp = self.qw(eps)  # samples from q(w)

                phig_s = self.phig(img_s)
                # w'*phi(G(x)) = (B x K)
                output = torch.mm(phig, wmode)
                s_output = torch.mm(phig_s, wmode)

                # w'*phi(G(x)) = (M x B x K)
                # wphig = torch.sum(
                #  wsamp.unsqueeze(1) * phig.unsqueeze(0).unsqueeze(3), dim=2 )

                # nll loss (equivalent to cross entropy loss)
                test_loss += criterion(output, label).item()

                # class prediction
                # pred = output.data.max(1)[1]  # n-dim {0,...,K-1}-valued
                pred = torch.argmax(output, dim=1)
                s_pred = torch.argmax(s_output, dim=1)
                # tensor.max(j) returns a list (A, B) where
                #   A = max of tensor over j-th dim
                #   B = argmax of tensor over j-th dim

                # corrects += pred.eq(label.data).cpu().numpy().sum()
                corrects += pred.eq(label).cpu().numpy().sum()
                s_corrects += s_pred.eq(label_s).cpu().numpy().sum()

                size += label.data.size()[0]
                s_size += label_s.data.size()[0]

        test_loss = test_loss / size

        prn_str = f'Epoch {epoch} Test set: Average nll loss: {test_loss: .4f}, ' \
                  f'Accuracy: {corrects}/{size} ({100. * corrects / size : .6f}), ' \
                  f'S_accuracy: {100 * s_corrects / s_size: .4f}%'
        print(prn_str)
        # save (append) the test scores/stats to files
        if record_file:
            record = open(record_file, 'a')
            print('recording %s\n' % record_file)
            record.write('%s\n' % (prn_str,))
            record.close()

        # save the models as files
        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.phig, '%s/model_epoch%s_phig.pt' % (self.checkpoint_dir, epoch))
            torch.save(self.qw, '%s/model_epoch%s_qw.pt' % (self.checkpoint_dir, epoch))
