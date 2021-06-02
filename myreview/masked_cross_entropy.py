import torch
from torch.nn import functional
from torch.autograd import Variable

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def masked_cross_entropy(logits, target, mask):
    """
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # print(sum(logits_flat))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # print(sum(log_probs_flat))
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # print(sum(target_flat))
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # print(sum(losses_flat))
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # print(sum(losses))
    # mask: (batch, max_len)
    loss = losses.masked_select(mask)
    # print(loss)
    # loss = losses
    # print(sum(loss))
    if len(loss) > 0:
        loss = loss.mean()
    else:
        loss = Variable(torch.zeros(1))
        loss = loss.cuda()
    
    return loss
