import sys
import torch
import math
import numpy as np
from uberduck_ml_dev.models.common import STFT
from uberduck_ml_dev.vocoders.istftnet import iSTFTNetGenerator, TorchSTFT

class Denoiser(torch.nn.Module):
    """WaveGlow denoiser, adapted for HiFi-GAN and iSTFTNet"""

    # Explicitly use cuda:0 to avoid the "cuda:None" error
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __init__(
        self, hifigan, filter_length=1024, n_overlap=4, win_length=1024, mode="zeros"
    ):
        super(Denoiser, self).__init__()
        self.stft = STFT(
            filter_length=filter_length,
            hop_length=int(filter_length / n_overlap),
            win_length=win_length,
            device=Denoiser.device
        )

        # Manual buffer migration for STFT (fixes "AttributeError: .to")
        if hasattr(self.stft, 'forward_basis'):
            self.stft.forward_basis = self.stft.forward_basis.to(Denoiser.device)
        if hasattr(self.stft, 'inverse_basis'):
            self.stft.inverse_basis = self.stft.inverse_basis.to(Denoiser.device)

        if mode == "zeros":
            mel_input = torch.zeros((1, 80, 88)).to(Denoiser.device)
        elif mode == "normal":
            mel_input = torch.randn((1, 80, 88)).to(Denoiser.device)
        elif mode == "ones":
            mel_input = torch.ones((1, 80, 88)).to(Denoiser.device)
        elif mode == "half":
            mel_input = torch.full((1, 80, 88), 0.5).to(Denoiser.device)
        elif mode == "normal_alt":
            mel_input = torch.rand((1, 80, 88)).to(Denoiser.device)
        elif mode == "sinusoid":
            t = torch.linspace(0, 2 * math.pi, steps=88)
            wave = torch.sin(t)
            mel_input = wave.repeat(80, 1).unsqueeze(0).to(Denoiser.device)
        else:
            raise Exception("Mode {} is not supported".format(mode))

        with torch.no_grad():
            if isinstance(hifigan, iSTFTNetGenerator):
                # Re-initialize for iSTFTNet specific settings
                self.stft = TorchSTFT(filter_length=16, hop_length=4, win_length=16, device=Denoiser.device)
                
                # Manual buffer migration for TorchSTFT window
                if hasattr(self.stft, 'window'):
                    self.stft.window = self.stft.window.to(Denoiser.device)
                
                # Keep spec and phase on GPU (don't use .cpu())
                spec, phase = hifigan.vocoder(mel_input)
                y_g_hat = self.stft.inverse(spec, phase)
                bias_audio = y_g_hat.view(1, -1).float()
            else:
                bias_audio = (
                    hifigan.vocoder.forward(mel_input)
                    .view(1, -1)
                    .float()
                )
            
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer("bias_spec", bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=10):
        """
        :param audio: Audio data tensor
        :param strength: Amount of bias removal (Recommended 10 - 50)
        :return: Denoised audio tensor
        """
        audio = audio.to(Denoiser.device).float()

        # Ensure transform and inverse operations stay on the same device
        audio_spec, audio_angles = self.stft.transform(audio)
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised
