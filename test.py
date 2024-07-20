import soundfile

from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import soundfile as sf


model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "Stars are like the memories of the universe, each one a glimpse into its past. The mind is a canvas, and thoughts are the brushstrokes that paint our reality. In the symphony of life, each experience plays a note that resonates through eternity. Time is a river, and we are the stones that shape its flow. The dance of chaos and order births the cosmos. A smile is the gravitational pull of happiness on the fabric of existence. Echoes remind us that even sound leaves a piece of itself behind."

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

# Convert the output to NumPy array
output_array = output.cpu().numpy()

# Save the waveform as a .wav file
sf.write("output.wav", data=output_array[0], samplerate=model.config.sampling_rate)
# scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output.float().numpy())

# data, samplerate = soundfile.read(audio_file)
# soundfile.write(audio_file, data, samplerate)






