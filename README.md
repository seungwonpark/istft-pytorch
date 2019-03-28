# istft-pytorch
Two different PyTorch implementation of Inverse-STFT.

Since there's no official implementation of Inverse-STFT in pytorch, people are trying to implement it by their own.
https://github.com/keunwoochoi/torchaudio-contrib/issues/27 was one of that, and this repository shows the
speed performance comparison of two different codes.

Speed comparison results are shown in [inspection.ipynb](./inspection.ipynb).
`istft_irfft.py` is much faster.

Please refer to https://github.com/keunwoochoi/torchaudio-contrib/issues/27 for further discussion.

## License & Authors
[Keunwoo Choi](https://github.com/keunwoochoi) initially implemented `istft_irfft.py`.
`istft_deconv.py` was implemented by me([Seungwon Park](https://github.com/seungwonpark)).

License: GNU General Public License v3.0
