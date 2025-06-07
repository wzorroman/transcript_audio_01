import os
from pydub import AudioSegment
import torch
import time
from faster_whisper import WhisperModel
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import sys
from pathlib import Path

# Añade la ruta del proyecto al PYTHONPATH -- Temporalmente
project_root = Path(__file__).parent.parent  # Sube dos niveles (a proy_videos_youtube/)
sys.path.append(str(project_root))
from helpers.handle_error import logger, log_decorator, timer


# ======================
# Funciones auxiliares
# ======================

@log_decorator()
def convert_mp3_to_wav(input_file: str, output_file: str = "converted_audio.wav") -> str:
    """
    Convierte un archivo MP3 a WAV mono a 16kHz (requisito de Whisper).
    """
    logger.info("Convirtiendo audio a formato WAV mono...")
    audio = AudioSegment.from_mp3(input_file)
    audio = audio.set_frame_rate(16000).set_channels(1)  # Whisper requiere 16kHz y mono
    audio.export(output_file, format="wav")
    logger.info(f"✅ Conversión completada: {output_file}")
    return output_file

@timer
def split_audio(file_path: str, chunk_length_ms: int = 300000, output_dir: str = "chunks") -> list:
    """
    Divide un archivo WAV en segmentos manejables (por defecto cada 5 minutos).
    Devuelve la lista de rutas de los segmentos.
    """
    chunks = []
    logger.info(f"Dividiendo audio en segmentos de {chunk_length_ms // 60000} minutos...")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.critical(f"Error al crear carpeta CHUNKS de salida: {output_dir} | {e}")
        return chunks
    
    try:
        audio = AudioSegment.from_wav(file_path)
    except Exception as e:
        logger.critical(f"Error cargando WAV: {file_path} | {e}")
        return chunks

    for i, start_time in enumerate(range(0, len(audio), chunk_length_ms)):
        chunk = audio[start_time:start_time + chunk_length_ms]
        chunk_name = os.path.join(output_dir, f"chunk_{i}.wav")
        chunk.export(chunk_name, format="wav")
        chunks.append(chunk_name)

    logger.info(f"✅ Se crearon {len(chunks)} segmentos.")
    return chunks


def transcribe_chunks(chunks: list, model_size: str = "medium", language: str = "es") -> list:
    """
    Transcribe una lista de archivos de audio usando Whisper.
    Devuelve una lista con las transcripciones por segmento.
    """
    logger.info(f"Cargando modelo Whisper ({model_size})...")
    model = whisper.load_model(model_size)

    full_transcriptions = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Transcribiendo segmento {i+1}/{len(chunks)}: {chunk}")
        try:
            result = model.transcribe(chunk, language=language)
        except Exception as e:
            logger.warning(f"ERROR segmento:{i+1}:{chunk} || {e}", exc_info=True)
            continue
        full_transcriptions.append(result["text"])

    logger.info("✅ Transcripción completada.")
    return full_transcriptions


def save_transcription(transcriptions: list, output_file: str = "transcripcion_completa.txt"):
    """
    Guarda todas las transcripciones en un único archivo de texto.
    """
    logger.info(f"Guardando transcripción en '{output_file}'...")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, text in enumerate(transcriptions):
                f.write(f"\n--- Segmento {idx+1} ---\n")
                f.write(text + "\n")
    except Exception as e:
        logger.critical(f"Error guardar transcripcion: {e}")

    logger.info(f"✅ Transcripción guardada en '{output_file}'")


# ======================
# multi
# ======================

def load_chunks_from_folder(chunk_dir: str = "chunks", extension: str = ".wav") -> list:
    """
    Carga todos los archivos de chunk desde una carpeta dada,
    filtra por extensión (por defecto .wav) y devuelve una lista ordenada
    con las rutas completas de los archivos.

    Args:
        chunk_dir (str): Ruta de la carpeta donde están los chunks.
        extension (str): Extensión de los archivos a buscar (ej: .wav).

    Returns:
        list: Lista ordenada con las rutas completas de los chunks.
    """
    if not os.path.isdir(chunk_dir):
        logger.critical(f"[ERROR] La carpeta '{chunk_dir}' no existe.")
        return []

    # Filtrar archivos con la extensión deseada y ordenar por nombre
    try:
        archivos_en_carpeta = os.listdir(chunk_dir)
        archivos_filtrados = [f for f in archivos_en_carpeta if f.endswith(extension)]
        rutas_completas = [os.path.join(chunk_dir, item) for item in archivos_filtrados]
        
        def get_number(ruta):
            name_tmp = os.path.splitext(os.path.basename(ruta))[0]
            return int(name_tmp.replace("chunk_", ""))

        chunk_files = sorted(rutas_completas, key=get_number)
        # chunk_files = sorted(
        #     [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(extension)],
        #     key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("chunk_", ""))
        # )
    except Exception as e:
        logger.critical(f"[ERROR] Error al leer o ordenar los archivos: {e}", exc_info=True)
        return []

    logger.info(f"✅ Se encontraron {len(chunk_files)} chunks en '{chunk_dir}'.")
    return chunk_files

def get_chunk_duration_ms(chunk_path):
    audio = AudioSegment.from_wav(chunk_path)
    return len(audio)

@log_decorator()
def transcribe_single_chunk_with_model2(args):
    """
    Función auxiliar para transcribir un único chunk.
    Diseñada para ser usada en paralelo.
    """
    chunk_path, model, language, output_dir, device = args
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configurar opciones de transcripción
    transcribe_options = {
        "language": language,
        "fp16": False if device == "cpu" else True  # FP16 solo para GPU
    }
    logger.info(f" OPC transcripcion => {transcribe_options}")
    logger.info(f"Chunk {chunk_path} - Duración: {get_chunk_duration_ms(chunk_path)} ms")
    try:
        logger.info(f" * Transcribiendo: {chunk_path}")
        #result = model.transcribe(chunk_path, language=language)
        time.sleep(1)
        result = model.transcribe(chunk_path, **transcribe_options)
        text = result["text"]
        logger.info(f" *   transcripcion => cant caracteres: {len(text)}")
        # Generar nombre del archivo de salida
        filename = os.path.splitext(os.path.basename(chunk_path))[0] + ".txt"
        output_path = os.path.join(output_dir, filename)

        # Guardar texto
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f" * Transcripcion finalizada: {chunk_path} | {output_path}")
        return output_path  # Devolver la ruta del archivo guardado
    except Exception as e:
        logger.warning(f"Error al transcribir {chunk_path}: {e}")
        return None

def transcribe_single_chunk_with_model(
    chunk_path, 
    model, 
    output_dir, 
    options,
    max_retries: int = 3,
    retry_delay: float = 1.0
):
    """
    Transcribe un único chunk reutilizando un modelo ya cargado.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f" * Transcribiendo: {chunk_path}")
            segments, info = model.transcribe(chunk_path, **options)
            full_text = " ".join(segment.text for segment in segments)
            logger.info(f" *   transcripcion => cant caracteres: {len(full_text)}")

            filename = os.path.splitext(os.path.basename(chunk_path))[0] + ".txt"
            output_path = os.path.join(output_dir, filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text.strip())

            logger.info(f" * Transcripcion finalizada: {chunk_path} | {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"❌ Error en intento {attempt} al transcribir {chunk_path}: {e}", exc_info=True)
            if attempt < max_retries:
                logger.info(f"🔄 Reintentando en {retry_delay} segundos...")
                time.sleep(retry_delay)
            else:
                logger.error(f"🛑 Chunk fallido tras {max_retries} intentos: {chunk_path}")
                return None
    
def transcribe_chunks_parallel_and_save(
    chunks: list,
    output_dir: str = "transcripcion_chunks",
    model_size: str = "base",
    language: str = "es",
    device: str = "cpu",
    max_workers: int = None
) -> list:
    """
    Transcribe chunks en paralelo y guarda cada uno individualmente.
    
    Args:
        chunks (list): Lista de rutas de archivos WAV.
        output_dir (str): Carpeta donde guardar las transcripciones individuales.
        model_size (str): Tamaño del modelo Whisper.
        language (str): Idioma del audio.
        max_workers (int): Máximo número de procesos simultáneos.

    Returns:
        list: Lista de rutas de archivos de transcripciones guardados.
    """
    os.makedirs(output_dir, exist_ok=True)
    # num_processes = max_workers or min(4, os.cpu_count())
    num_threads = max_workers or min(4, os.cpu_count())

    try:
        start_load = time.time()
        model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
        print(f"({device}) Modelo [{model_size}] cargado: {time.time() - start_load:.2f} seg")    
    except Exception as e:
        logger.warning(f"Error al cargar modelo: {e}")
        return []
        
    logger.info(f"Iniciando transcripción paralela con {num_threads} procesos...")
    
    # Configurar opciones comunes de transcripción
    transcribe_options = {
        "language": language,
        "without_timestamps": True,
        "max_new_tokens": 128,
        "beam_size": 5
    }
    
    # Función auxiliar que recibe solo el chunk
    def wrapper(chunk):
        return transcribe_single_chunk_with_model(chunk, model, output_dir, transcribe_options)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        saved_files = list(executor.map(wrapper, chunks))

    # Filtrar resultados nulos (chunks que fallaron)
    saved_files = [f for f in saved_files if f is not None]
    logger.info(f"✅ Se guardaron {len(saved_files)} transcripciones individuales.")
    return saved_files


def get_unprocessed_chunks(chunks: list, output_dir: str = "trans_chunks") -> list:
    unprocessed = []
    for chunk in chunks:
        base_name = os.path.splitext(os.path.basename(chunk))[0] + ".txt"
        output_path = os.path.join(output_dir, base_name)
        if not os.path.exists(output_path):
            unprocessed.append(chunk)
        else:
            logger.info(f"chunk ya procesado anteriormente : {chunk}")
            continue
        
    return unprocessed


def merge_transcriptions(file_paths: list, output_file: str = "transcripcion_completa.txt"):
    """
    Une varias transcripciones individuales en un solo archivo final.
    """
    logger.info(f"Uniendo {len(file_paths)} archivos de transcripción...")

    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            for idx, file_path in enumerate(sorted(file_paths)):
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    #outfile.write(f"\n--- Segmento {idx+1} ---\n")
                    outfile.write(content + "\n")
        logger.info(f"✅ Transcripción completa guardada en '{output_file}'")
    except Exception as e:
        logger.critical(f"Error al unir transcripciones: {e}")
            
# ======================
# Función principal
# ======================

@log_decorator()
def main_v1():
    
    input_file = "outputs/audio.mp3"
    output_file_wav = "outputs/audio_converted.wav"
    output_file_txt = "outputs/transcripcion.txt"
    output_dir_chunks = "outputs/chunks"
    
    split_in_min = 1
    CHUNKS_LENGTH_SIZE = int(split_in_min * 60 * 1000)  # minutos → segundos → ms
    # CHUNKS_LENGTH_SIZE = 300000 # Segmentos de 5 minutos
    
    wav_file = convert_mp3_to_wav(input_file, output_file_wav)
    chunks = split_audio(wav_file, chunk_length_ms=CHUNKS_LENGTH_SIZE, output_dir=output_dir_chunks)
    # transcriptions = transcribe_chunks(chunks, model_size="medium", language="es")
    # save_transcription(transcriptions, output_file_txt)

@log_decorator()
def main_v2():
    # procesa desde los chunks ya almacenados en una carpeta
    output_dir_chunks = "outputs/chunks_min_2"
    output_dir_transcriptions = "outputs/transcriptions"
    
    chunks = load_chunks_from_folder(output_dir_chunks)
    # verificar los chunks ya procesados
    chunks = get_unprocessed_chunks(chunks)
    # 👇 Transcribir y guardar cada chunk individualmente
    # model_size_opc = tiny, base o medium
    saved_files = transcribe_chunks_parallel_and_save(
        chunks,
        output_dir=output_dir_transcriptions,
        model_size="tiny",
        language="es",
        max_workers=2
    )

    # 👇 Unir todas las transcripciones en un solo archivo
    merge_transcriptions(saved_files, output_file="transcripcion_completa.txt")


if __name__ == "__main__":
    main_v2()
    