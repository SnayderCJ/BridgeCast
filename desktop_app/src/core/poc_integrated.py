import sounddevice as sd
import numpy as np
import queue
import sys
import argparse
from stt_engine import STTEngine
from translator import TranslatorEngine
from tts_engine import TTSEngine

# Configuración de audio
SAMPLING_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 3
BLOCK_SIZE = int(SAMPLING_RATE * BLOCK_DURATION)

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, help="ID del dispositivo de entrada", default=15)
    args = parser.parse_args()

    print("--- BridgeCast: Prueba de Concepto (STT + TRADUCCIÓN + TTS) ---")
    
    try:
        # Inicializar todos los motores
        stt = STTEngine(model_size="tiny")
        translator = TranslatorEngine(source_lang="es", target_lang="en")
        tts = TTSEngine(language="en") # Voz en inglés para la salida
    except Exception as e:
        print(f"Error cargando los modelos: {e}")
        return

    try:
        device_info = sd.query_devices(args.device, 'input')
        print(f"\nCapturando desde: {device_info['name']} (ID: {args.device})")
    except Exception as e:
        print(f"Error al acceder al dispositivo {args.device}: {e}")
        return

    print("Habla algo en Español y espera la voz traducida (Presiona Ctrl+C para detener)...\n")

    try:
        with sd.InputStream(device=args.device,
                            samplerate=SAMPLING_RATE, 
                            channels=CHANNELS, 
                            callback=audio_callback,
                            blocksize=BLOCK_SIZE):
            while True:
                audio_chunk = audio_queue.get()
                audio_data = audio_chunk.flatten().astype(np.float32)
                
                # 1. Transcribir (STT)
                text, lang = stt.transcribe(audio_data)
                
                if text:
                    # 2. Traducir (Traducción)
                    translated_text = translator.translate(text)
                    print(f"\n--- [Procesado] ---")
                    print(f"[ES] Original:  {text}")
                    print(f"[EN] Traducido: {translated_text}")
                    
                    # 3. Sintetizar y Reproducir (TTS)
                    print(f"[Voz] Generando audio...")
                    tts.generate_and_play(translated_text)
                    print(f"------------------")
                else:
                    pass

    except KeyboardInterrupt:
        print("\nPrueba finalizada por el usuario.")
    except Exception as e:
        print(f"\nOcurrió un error durante la ejecución: {e}")

if __name__ == "__main__":
    main()
