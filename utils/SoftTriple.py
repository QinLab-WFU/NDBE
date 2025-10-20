# Implementation of SoftTriple Loss
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K#4
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        #marginM = torch.zeros(simClass.shape).cuda()
        #marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        #marginM[target_multi==1] = self.margin
        mask=torch.zeros_like(simClass)
        #mask.scatter_(1,target.view(-1,1),1)
        marginM=self.margin*mask
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify
class RelaHashLoss(nn.Module):
    def __init__(self,
                 beta=10,
                 m=0.1,
                 multiclass=False,
                 onehot=True,
                 **kwargs):
        super(RelaHashLoss, self).__init__()
        self.beta = beta
        self.m = m
        self.multiclass = multiclass
        self.onehot = onehot

    def compute_margin_logits(self, logits, labels):
        if self.multiclass:
            y_onehot = labels * self.m
            # print(y_onehot)
            margin_logits = self.beta * (logits - y_onehot)
        else:
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
            margin_logits = self.beta * (logits - y_onehot)
        return margin_logits

    def forward(self, logits,  labels):
        if self.multiclass:
            if not self.onehot:
                labels = F.one_hot(labels, logits.size(1))

            labels = labels.float()


            margin_logits = self.compute_margin_logits(logits, labels)

            log_logits = F.log_softmax(margin_logits, dim=1)
            # t2 = margin_logits.cpu().detach().numpy()
            # np.savetxt('log_logits.csv', t2, delimiter=',')
            # lable_num = labels.shape[1]
            A = ((labels==0).sum(dim=1) == labels.shape[1])
            labels[A==True] = 1
            labels_scaled = labels / labels.sum(dim=1, keepdim=True)
            # t3 = labels_scaled.cpu().detach().numpy()
            # np.savetxt('labels_scale.csv', t3, delimiter=',')
            loss = - (labels_scaled * log_logits).sum(dim=1)
            # print(loss)
            # t1 = loss.cpu().detach().numpy()
            # np.savetxt('loss.csv',t1,delimiter=',')
            loss = loss.mean()
        else:
            if self.onehot:
                labels = labels.argmax(1)

            margin_logits = self.compute_margin_logits(logits, labels)
            loss = F.cross_entropy(margin_logits, labels)
        return loss