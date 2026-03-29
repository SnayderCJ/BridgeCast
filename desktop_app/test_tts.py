import wave
import struct
import numpy as np
from src.core.tts_engine import TTSEngine

tts = TTSEngine()
audio = tts.model.apply_tts(text='Hello, this is a test from BridgeCast.', speaker='en_0', sample_rate=tts.sample_rate).numpy()

# Convert to 16-bit PCM
audio_int16 = np.int16(audio * 32767)

with wave.open('test.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(tts.sample_rate)
    f.writeframes(audio_int16.tobytes())

print("Archivo test.wav generado.")
