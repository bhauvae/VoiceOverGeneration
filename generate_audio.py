import soundfile


def generate_audio(video_object):

    audio_file = video_object.audio_file

    text = video_object.quote

    data, samplerate = soundfile.read(audio_file)
    soundfile.write(audio_file, data, samplerate)


