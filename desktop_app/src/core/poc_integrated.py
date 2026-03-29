import sounddevice as sd
import numpy as np
import queue
import sys
import argparse
from stt_engine import STTEngine
from translator import TranslatorEngine
from tts_engine import TTSEngine

# Configuración técnica avanzada
SAMPLING_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.3 # Bloques más pequeños para mayor agilidad
BLOCK_SIZE = int(SAMPLING_RATE * BLOCK_DURATION)

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, help="ID del dispositivo", default=15)
    parser.add_argument("--virtual-mic", action="store_true", help="Usar micrófono virtual")
    args = parser.parse_args()

    print("--- BridgeCast Ultra: Nivel Profesional ---")
    
    try:
        # Cargamos MEDIUM para el equilibrio perfecto entre precisión y velocidad
        stt = STTEngine(model_size="medium") 
        translator = TranslatorEngine(source_lang="es", target_lang="en")
        tts = TTSEngine(language="en")
    except Exception as e:
        print(f"Error: {e}")
        return

    audio_buffer = []
    silence_blocks = 0
    # Umbrales afinados para ignorar ruido de fondo de reuniones
    MIN_VOICE_RMS = 0.04 
    SILENCE_LIMIT = 3 # Aprox 0.9 segundos de silencio para procesar

    print("\n[LISTO] Escuchando con precisión de grado reunión...")

    try:
        with sd.InputStream(device=args.device,
                            samplerate=SAMPLING_RATE, 
                            channels=CHANNELS, 
                            callback=audio_callback,
                            blocksize=BLOCK_SIZE):
            while True:
                audio_chunk = audio_queue.get()
                audio_data = audio_chunk.flatten().astype(np.float32)
                
                rms = np.sqrt(np.mean(audio_data**2))
                
                if rms > MIN_VOICE_RMS:
                    # Voz activa: acumulamos
                    audio_buffer.append(audio_data)
                    silence_blocks = 0
                    print("🎤", end="", flush=True)
                else:
                    # Silencio o ruido: comprobamos si hay frase pendiente
                    if audio_buffer:
                        silence_blocks += 1
                        if silence_blocks >= SILENCE_LIMIT:
                            print("\n[Procesando con alta fidelidad...]")
                            
                            full_audio = np.concatenate(audio_buffer)
                            audio_buffer = [] 
                            silence_blocks = 0
                            
                            text, _ = stt.transcribe(full_audio, language="es")
                            
                            if text and len(text) > 3:
                                # Filtro básico para eliminar transcripciones de "ruido" que parecen texto
                                if text.lower() in ["transcripción precisa.", "continuará", "gracias.", "bye."]:
                                    print("[Ruido de fondo ignorado]")
                                    continue
                                    
                                print(f"-> [ES] {text}")
                                translated = translator.translate(text)
                                print(f"-> [EN] {translated}")
                                
                                # Reproducción inmediata
                                tts.generate_and_play(translated, virtual_mic=args.virtual_mic)
                            else:
                                print("[Ruido sutil descartado]")

    except KeyboardInterrupt:
        print("\nMotor detenido.")
    except Exception as e:
        print(f"\nError crítico: {e}")

if __name__ == "__main__":
    main()
