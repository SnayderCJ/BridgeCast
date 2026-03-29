import os
import torch
import sounddevice as sd
import numpy as np

class TTSEngine:
    def __init__(self, language="en", speaker="v3_en"):
        """
        Inicializa Silero TTS.
        language: 'en' o 'es'
        speaker: El modelo de voz específico.
        """
        self.device = torch.device("cpu")
        self.sample_rate = 48000
        
        # Ruta del modelo (se descargará automáticamente)
        model_url = f"https://models.silero.ai/models/tts/{language}/{speaker}.pt"
        model_path = os.path.join(os.getcwd(), f"model_{language}.pt")
        
        print(f"Cargando motor TTS para idioma: {language}...")
        
        if not os.path.exists(model_path):
            print("Descargando modelo TTS (esto solo ocurre una vez)...")
            torch.hub.download_url_to_file(model_url, model_path)
        
        self.model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
        self.model.to(self.device)
        print("Motor TTS listo.")

    def generate_and_play(self, text, speaker="en_0", virtual_mic=False):
        """
        Genera audio a partir de texto y lo reproduce inmediatamente.
        Si virtual_mic=True, envía el audio al micrófono virtual de PulseAudio.
        """
        if not text.strip():
            return

        # Generar audio
        audio = self.model.apply_tts(text=text,
                                    speaker=speaker,
                                    sample_rate=self.sample_rate)
        
        # Convertir a numpy para reproducir con sounddevice
        audio_numpy = audio.numpy()
        
        # Configurar salida
        if virtual_mic:
            os.environ["PULSE_SINK"] = "BridgeCastVirtualMic"
            device_name = "pulse"
        else:
            if "PULSE_SINK" in os.environ:
                del os.environ["PULSE_SINK"]
            device_name = "pulse" # Usar pulse por defecto para evitar problemas con ALSA

        # Reproducir
        sd.play(audio_numpy, self.sample_rate, device=device_name)
        # No bloqueamos, permitimos que siga el proceso
        # sd.wait() # Si quisiéramos esperar a que termine de hablar

if __name__ == "__main__":
    # Prueba rápida
    tts = TTSEngine()
    tts.generate_and_play("Hello, this is a test of the BridgeCast translation system.")
    sd.wait() # Esperar a que termine la prueba
