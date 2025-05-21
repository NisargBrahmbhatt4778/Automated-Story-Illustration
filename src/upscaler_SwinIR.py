import sys
import os

# Add the models directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))

import torch
from network_swinir import SwinIR

def setup_swinir_upscaler():
    """
    Set up the SwinIR upscaler with the correct configuration for the pretrained model.
    """

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Match the model config for the SwinIR-M x4 model
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='1conv'
    )


    # Load pretrained checkpoint
    model_path = '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
    ckpt = torch.load(model_path, map_location=device)

    # Fix: extract 'params_ema' if present
    if isinstance(ckpt, dict) and 'params_ema' in ckpt:
        ckpt = ckpt['params_ema']

    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model = model.to(device)

    return model, device
