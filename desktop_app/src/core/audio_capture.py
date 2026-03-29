import sounddevice as sd

def list_audio_devices():
    """Lista todos los dispositivos de audio disponibles en el sistema."""
    print("--- Dispositivos de Audio Detectados ---")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        kind = ""
        if device['max_input_channels'] > 0:
            kind = "[ENTRADA]"
        if device['max_output_channels'] > 0:
            kind += " [SALIDA]"
        
        print(f"{i}: {device['name']} {kind} (Canales: {device['max_input_channels']} In / {device['max_output_channels']} Out)")
    print("---------------------------------------")

if __name__ == "__main__":
    list_audio_devices()
