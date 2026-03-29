from faster_whisper import WhisperModel
import numpy as np

class STTEngine:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Inicializa el motor Whisper optimizado para CPU.
        compute_type="int8" reduce el consumo de RAM y aumenta la velocidad en CPU.
        """
        print(f"Cargando modelo STT ({model_size}) en {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Modelo cargado correctamente.")

    def transcribe(self, audio_data: np.ndarray):
        """
        Transcribe un chunk de audio a texto.
        audio_data debe ser un array de numpy (float32, 16kHz).
        """
        segments, info = self.model.transcribe(audio_data, beam_size=5)
        text = "".join([segment.text for segment in segments])
        return text.strip(), info.language

if __name__ == "__main__":
    # Prueba rápida de carga del modelo
    stt = STTEngine()
    print("Motor STT listo para pruebas.")
