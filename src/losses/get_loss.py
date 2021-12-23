import torch
from src.losses.fft_loss import FFTLoss


def get_loss(loss_name: str = 'L1'):
    if loss_name == 'L1':
        return torch.nn.L1Loss()
    elif loss_name == 'fft':
        return FFTLoss()
