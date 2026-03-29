from transformers import MarianMTModel, MarianTokenizer
import torch

class TranslatorEngine:
    def __init__(self, source_lang="es", target_lang="en"):
        """
        Inicializa el modelo de traducción de Helsinki-NLP.
        Ejemplo: source_lang="es", target_lang="en" para Español -> Inglés.
        """
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        print(f"Cargando modelo de traducción ({model_name})...")
        
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        
        # Mover a CPU (especificado en los requisitos)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        print("Modelo de traducción listo.")

    def translate(self, text):
        """Traduce una cadena de texto."""
        if not text.strip():
            return ""
            
        # Preparar el texto para el modelo
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # Generar traducción
        with torch.no_grad():
            translated_tokens = self.model.generate(**inputs)
            
        # Decodificar el resultado
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

if __name__ == "__main__":
    # Prueba rápida
    translator = TranslatorEngine()
    test_text = "Hola, esto es una prueba de traducción en tiempo real."
    result = translator.translate(test_text)
    print(f"Original: {test_text}")
    print(f"Traducido: {result}")
