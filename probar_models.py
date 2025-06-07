import torch
import whisper
import time


chunk_path = "outputs/chunks/chunk_2.wav"
output_path = "transcripcion_demo_base.txt"
modelo_cargado = "base" # tiny, base o medium


device = "cuda" if torch.cuda.is_available() else "cpu"

# Carga el modelo (prueba con "tiny", "base" o "medium")
start_load = time.time()
model = whisper.load_model(modelo_cargado)
end_load = time.time()
print(f"Modelo [{modelo_cargado}] cargado en {end_load - start_load:.2f} segundos")    

# Configurar opciones de transcripción
transcribe_options = {
    "language": "es",
    "fp16": False if device == "cpu" else True  # FP16 solo para GPU
}
    
# Transcribe
start_trans = time.time()
result = model.transcribe(chunk_path, **transcribe_options)
end_trans = time.time()
print(f"Transcrito en {end_trans - start_trans:.2f} segundos")

try:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
except Exception as e:
    print(f"Error: {e}")
    
print(result["text"])

# Modelo [tiny] cargado en 5.19 seg | Transcrito en 180.32 seg
