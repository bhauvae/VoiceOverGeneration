#%%
import soundfile

#%%
mixwav_mc, sr = soundfile.read(r"C:\Users\Bhavya\Projects\Python Projects\ShortForm-VideoCreationPipeline\test4.wav")
# mixwav.shape: num_samples, num_channels
mixwav_sc = mixwav_mc[:]
print(mixwav_sc)
#%%
from espnet_model_zoo.downloader import ModelDownloader

d = ModelDownloader()
cfg = d.download_and_unpack("espnet/Wangyou_Zhang_chime4_enh_train_enh_conv_tasnet_raw")
# %%
import sys
import soundfile
from espnet2.bin.enh_inference import SeparateSpeech


separate_speech = {}
# For models downloaded from GoogleDrive, you can use the following script:
enh_model_sc = SeparateSpeech(
  train_config=cfg["train_config"],
  model_file=cfg["model_file"],
  # for segment-wise process on long speech
  normalize_segment_scale=False,
  show_progressbar=True,
  ref_channel=4,
  normalize_output_wav=True,
)
# %%
from IPython.display import display, Audio
wave = enh_model_sc(mixwav_sc[None, ...], sr)
print("Input real noisy speech", flush=True)

print("Enhanced speech", flush=True)
display(Audio(wave[0].squeeze(), rate=sr))
# %%
