**IMPORTANT:** `torch.istft` will be available from `torch>=1.6.0`. See [pytorch/pytorch#35569](https://github.com/pytorch/pytorch/pull/35569)

# istft-pytorch
Two different PyTorch implementation of Inverse-STFT.

Since there's no official implementation of Inverse-STFT in pytorch, people are trying to implement it by their own.
https://github.com/keunwoochoi/torchaudio-contrib/issues/27 was one of that, and this repository shows the
speed performance comparison of two different codes.

Speed comparison results are shown in https://github.com/seungwonpark/istft-pytorch/issues/3.
`istft_deconv.py` is much faster.

Please refer to https://github.com/keunwoochoi/torchaudio-contrib/issues/27 for further discussion.

## License & Authors
[Keunwoo Choi](https://github.com/keunwoochoi) initially implemented `istft_irfft.py`.
`istft_deconv.py` was implemented by me([Seungwon Park](https://github.com/seungwonpark)).
`inspection.py` was kindly written by [Juan F. Montesinos](https://github.com/JuanFMontesinos).

License: GNU General Public License v3.0
