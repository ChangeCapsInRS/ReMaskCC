import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor

def accuracy(scores, targets, k):
    
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)

def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
    score_dict = dict(zip(method, score))

    return score_dict

class DynamicTripletLoss(nn.Module):
    
    def __init__(self, margin=0.2):
        super(DynamicTripletLoss, self).__init__()
        self.margin = margin

    def cosine_sim(self, x1, x2):
        return F.cosine_similarity(x1, x2, dim=1)

    def forward(self, d12, d21):

        pos_sim = self.cosine_sim(d12, d21)
        
        d21_neg = torch.roll(d21, shifts=1, dims=0)
        d12_neg = torch.roll(d12, shifts=1, dims=0)
        
        neg_sim_1 = self.cosine_sim(d12, d21_neg)
        neg_sim_2 = self.cosine_sim(d21, d12_neg)
        
        loss1 = torch.clamp(self.margin - pos_sim + neg_sim_1, min=0.0)
        loss2 = torch.clamp(self.margin - pos_sim + neg_sim_2, min=0.0)
        
        total_loss = torch.mean(loss1 + loss2)
        return total_loss