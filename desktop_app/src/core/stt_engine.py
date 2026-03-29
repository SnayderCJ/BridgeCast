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

    def transcribe(self, audio_data: np.ndarray, language="es"):
        """
        Transcribe audio con precisión profesional.
        Optimizado para ignorar ruido de fondo y evitar alucinaciones.
        """
        # Prompt mínimo y técnico para evitar que la IA lo repita por error
        prompt = "Transcripción precisa."
        
        segments, info = self.model.transcribe(
            audio_data, 
            beam_size=5, # Reducido ligeramente para ganar velocidad sin perder precisión
            language=language,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.6, # Más agresivo: solo captura voz clara
                min_speech_duration_ms=250,
                min_silence_duration_ms=400 # Pausa más corta para ganar velocidad
            )
        )
        
        text = "".join([segment.text for segment in segments])
        return text.strip(), info.language

if __name__ == "__main__":
    # Prueba rápida de carga del modelo
    stt = STTEngine()
    print("Motor STT listo para pruebas.")
