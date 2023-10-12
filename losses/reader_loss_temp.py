import torch
from torch.nn import CrossEntropyLoss


#TODO: dprabhu : This is the loss used by Span selection models (DPR) while training. Find appropriate structure to store this.
def get_loss(start_positions, end_positions, answer_mask, start_logits, end_logits, sel_logits, N, M):
    answer_mask = answer_mask.type(torch.FloatTensor).cuda()
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)
    loss_fct = CrossEntropyLoss(reduce=False, ignore_index=ignored_index)

    sel_logits = sel_logits.view(N, M)
    sel_labels = torch.zeros(N, dtype=torch.long).cuda()
    sel_loss = torch.sum(loss_fct(sel_logits, sel_labels))
    start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask) \
                    for (_start_positions, _span_mask) \
                    in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1))]
    end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) \
                    for (_end_positions, _span_mask) \
                    in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))]
    loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
        torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)
    loss_tensor=loss_tensor.view(N, M, -1).max(dim=1)[0]
    span_loss = _take_mml(loss_tensor)
    return span_loss + sel_loss

def _take_mml(loss_tensor):
    marginal_likelihood = torch.sum(torch.exp(
            - loss_tensor - 1e10 * (loss_tensor==0).float()), 1)
    return -torch.sum(torch.log(marginal_likelihood + \
                                torch.ones(loss_tensor.size(0)).cuda()*(marginal_likelihood==0).float()))