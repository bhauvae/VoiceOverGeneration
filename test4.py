from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from scipy.io.wavfile import read as read_wav
import nltk
import numpy as np
import soundfile


# download and load all models
preload_models()

# generate audio from text
text_prompt = "My name is Dhruv Gupta, and I - uh love biryani [laughs]"

sentences = nltk.sent_tokenize(text_prompt)
SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]

# save audio to disk

soundfile.write("test4-dhruv.wav", data=np.concatenate(pieces), samplerate=SAMPLE_RATE)
