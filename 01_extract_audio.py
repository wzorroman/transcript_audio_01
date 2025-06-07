from moviepy import VideoFileClip


def extract_audio(mp4_file, output_file_mp3):
    # Carga el videoclip
    video_clip = VideoFileClip(mp4_file) 

    # Extrae el audio del videoclip
    audio_clip = video_clip.audio 

    # Escribe el audio en un archivo separado
    audio_clip.write_audiofile(output_file_mp3) 

    # Cierra los clips de video y audio
    audio_clip.close() 
    video_clip.close() 
    print( "¡Extracción de audio exitosa!" )
    
# -------------------
if __name__ == "__main__":
    mp4_file = "videos/tema_01 - 02 Mayo - Viernes.mp4"
    mp3_file = "outputs/audio.mp3" 
    extract_audio(mp4_file, mp3_file)
