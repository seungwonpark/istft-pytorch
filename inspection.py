#!/usr/bin/env python
# coding: utf-8

# # Comparison of Inverse-STFT implementations
# - Seungwon Park's implementation: IFFT + deconvolution for stacking `ytmp`
# - Keunwoo Choi's implementation: based on IRFFT
# - Both are based on librosa's implementation
#   - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft

# In[1]:


import time
import torch
import librosa


# In[2]:


from istft_deconv import istft_deconv
from istft_irfft import istft_irfft


# In[3]:


y, sr = librosa.load(librosa.util.example_audio_file(), duration=2.0)
n_fft = 2048
hop_length = n_fft // 4
y = torch.tensor(y)
stft = torch.stft(y, n_fft, hop_length)


# In[4]:


stft_single = stft
stft_batch = stft.unsqueeze(0)


# In[5]:


result_deconv = istft_deconv(stft_single, hop_length)
result_irfft = istft_irfft(stft_batch, hop_length)[0]

diff = torch.max(torch.abs(result_deconv - result_irfft)).item()

if diff < 1e-4:
        print(f'Results are consistent. Maximum difference: {diff}')


        # In[6]:


        get_ipython().run_line_magic('timeit', 'result_deconv = istft_deconv(stft_single, hop_length)')


        # In[7]:


        get_ipython().run_line_magic('timeit', 'result_irfft = istft_irfft(stft_batch, hop_length)[0]')


        # # Conclusion
        # - IRFFT-based implementation is faster, showing that better parallelization doesn't outspeed algorithmic optimization.

