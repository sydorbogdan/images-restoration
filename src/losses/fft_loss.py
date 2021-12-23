import torch


class FFTLoss:
    def __init__(self):
        self.criterion = torch.nn.L1Loss()

    def forward(self, _input, target):
        print("Hi from fft loss")

        # l1 loss
        l1 = self.criterion(_input, target)

        # fft loss
        label_fft1 = torch.fft.rfft2(_input, norm="backward")
        pred_fft1 = torch.fft.rfft2(target, norm="backward")
        f = self.criterion(pred_fft1, label_fft1)

        # total loss
        loss = l1 + 0.2 * f

        return loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
