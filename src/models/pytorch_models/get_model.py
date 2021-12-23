from src.models.pytorch_models.UNet.unet_model import UNet
from src.models.pytorch_models.MIMOUNet.MIMOUNet import MIMOUNet
from src.models.pytorch_models.MIMOUNetFFT.MIMOUNetFFT import MIMOUNetFFT


def get_model(model_name: str = 'UNet', **kwargs):
    if model_name == 'UNet':
        return UNet(kwargs)
    elif model_name == 'MIMOUNet':
        return MIMOUNet(num_res=1)
    elif model_name == 'MIMOUNetFFT':
        return MIMOUNetFFT(num_res=1)
