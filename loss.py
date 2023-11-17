import torch
import torch.nn as nn

class KTLoss(nn.Module):

    def __init__(self):
        super(KTLoss, self).__init__()

    def forward(self, pred_answers, real_answers):
        real_answers = real_answers[:, 1:]
        answer_mask = torch.ne(real_answers, -1)
        pred_one, pred_zero = pred_answers.clone(), 1.0 - pred_answers

        pred_one[answer_mask] = torch.log(pred_one[answer_mask])
        pred_zero[answer_mask] = torch.log(pred_zero[answer_mask])
        pred = torch.cat((pred_zero.unsqueeze(dim = 1), pred_one.unsqueeze(dim = 1)), dim = 1)
        nll_loss = nn.NLLLoss(ignore_index = -1)
        loss = nll_loss(pred, real_answers.long())
        
        return loss