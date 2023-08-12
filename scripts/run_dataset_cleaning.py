import gin
from dataset.datasetAnalysis import DatasetAnalysis

if __name__ == '__main__':
    gin.parse_config_file("./automatic-speech-recognition-project/config/dataset_cleaning_config.gin")
    dataset_analysis = DatasetAnalysis()
    print("CONFIG PARSED. STARTING DATASET CLEANING...")
    dataset_analysis.remove_outliers()
    print("PROCESS FINISHED WITHOUT ERRORS")
