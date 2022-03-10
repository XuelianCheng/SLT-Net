import torch
import torch.nn.functional as F


def hybrid_e_loss(pred, mask):
    """
    Introduction: 
        Hybrid Eloss
    Paramaters:
        :param pred:
        :param mask:
        :return:
    Usage:
        loss = hybrid_e_loss(prediction_map, gt_mask)
        loss.backward()
    """
    # adaptive weighting masks
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)

    # weighted e loss function
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2], pred.shape[3])
    phiFM = pred - mpred

    mmask = mask.mean(dim=(2, 3)).view(mask.shape[0], mask.shape[1], 1, 1).repeat(1, 1, mask.shape[2], mask.shape[3])
    phiGT = mask - mmask

    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))

    # weighted iou loss function
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    return (wbce + eloss + wiou).mean()
