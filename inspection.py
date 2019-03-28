#!/usr/bin/env python
# coding: utf-8

# # Comparison of Inverse-STFT implementations
# - Seungwon Park's implementation: IFFT + deconvolution for stacking `ytmp`
# - Keunwoo Choi's implementation: based on IRFFT
# - Both are based on librosa's implementation
#   - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft


import torch
import librosa


from istft_deconv import istft_deconv
from istft_irfft import istft_irfft

def timing(func):
    def inner(*args,**kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = func(*args,**kwargs)
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        return output,time
    return inner
istft_irfft = timing(istft_irfft)
istft_deconv = timing(istft_deconv)
def test_stft():
    import traceback
    audio, sr = librosa.load(librosa.util.example_audio_file(), duration=2,sr=None)
    audio = torch.FloatTensor(audio)
    if torch.cuda.is_available():
        audio = audio.cuda()
    
    def mse(ground_truth, estimated):
        return torch.mean((ground_truth - estimated)**2)

    def to_np(tensor):
        return tensor.cpu().data.numpy()

    for i in range(1,12):
        filter_length = 2**i
        for j in range(i+1):
            try:
                hop_length = 2**j
                output = torch.stft(audio,n_fft =filter_length, hop_length=hop_length,window=torch.hann_window(round(filter_length)).cuda())
                result_deconv,time_deconv = istft_deconv(output,88200,hop_length=hop_length,win_length=filter_length)
                result_irff,time_irff = istft_irfft(output.unsqueeze(0),88200,hop_length=hop_length,win_length=filter_length)
                result_irff.squeeze_()
                loss_deconv = mse(result_deconv, audio)
                loss_irff = mse(result_deconv, audio)
                assert result_deconv.size()[-1] == audio.size()[-1]
                print('MSE [deconv: {0:.5f} / irff: {4:.5f} ]@ filter_length = {1}, hop_length = {2}, time_deconv = {3:.3f},time_irff = {5:.3f}'.format(to_np(loss_deconv), filter_length, hop_length,time_deconv,loss_irff,time_irff))
            except:
                pass
                print('Failed @ filter_length = {0}, hop_length = {1}, a:{2}, o {3}'.format(filter_length, hop_length,audio.size(),output.size()))
#                print(traceback.print_exc())







if __name__ == "__main__":
    test_stft()
=======
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

