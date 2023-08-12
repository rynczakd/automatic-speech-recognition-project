import gin
from audio_preparation.audioPreparation import AudioPreparation

if __name__ == '__main__':
    gin.parse_config_file("./automatic-speech-recognition-project/config/audio_preparation_config.gin")
    dataset_generator = AudioPreparation()
    print("CONFIG PARSED. STARTING GENERATION PROCESS...")
    dataset_generator.generate_database()
    print("PROCESS FINISHED WITHOUT ERRORS")
