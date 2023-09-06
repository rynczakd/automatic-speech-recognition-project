import gin
from procedures.train_asr import BaselineTraining

if __name__ == '__main__':
    gin.parse_config_file("./automatic-speech-recognition-project/config/asr_training_config.gin")
    baseline_training = BaselineTraining()
    print("CONFIG PARSED. TRAINING PROCESS STARTED...")
    baseline_training.train()
    print("PROCESS FINISHED WITHOUT ERRORS")
