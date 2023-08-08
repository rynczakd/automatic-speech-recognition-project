import os
import gin
import string


class CtcTokenizer:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    @gin.configurable
    def prepare_vocabulary(self, remove_punctuation: bool = False) -> (dict, dict):
        # Prepare empty list for .trans.txt files
        text_files = list()
        for sub_dir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".txt"):
                    text_files.append(os.path.join(sub_dir, file))

        # Prepare empty set for unique characters
        unique_characters = set()
        # Iterate over all lines for each text file
        for transcription in text_files:
            # This iteration allows us to get all the unique symbols present in the transcriptions of the dataset,
            # if we find symbols other than letters of the alphabet
            with open(transcription, "r") as t:
                line = t.readline()
                while line:
                    transcript = line.split(" ", 1)[1]
                    transcript = transcript.strip().lower().replace("\n", "")
                    if remove_punctuation:
                        transcript = transcript.translate(str.maketrans('', '', string.punctuation))
                    unique_characters.update(set(transcript))
                    line = t.readline()

        # Convert obtained set to a sorted list
        vocabulary = list(unique_characters)
        # Replace space symbol with <SPACE> for future processing
        vocabulary = sorted(list(map(lambda x: x.replace(' ', '<SPACE>'), vocabulary)))

        # Create character-to-index and index-to-character maps
        char_to_idx = {char: int(idx) for idx, char in enumerate(vocabulary, 1)}
        idx_to_char = {int(idx): char for idx, char in enumerate(vocabulary, 1)}

        # Replace <SPACE> with space symbol for future encoding
        idx_to_char = {int(idx): char.replace('<SPACE>', ' ') if char == '<SPACE>' else char
                       for idx, char in idx_to_char.items()}

        return char_to_idx, idx_to_char

    @staticmethod
    def tokenizer(vocabulary: dict, sentence: string) -> list:
        tokens = [vocabulary['<SPACE>'] if char == ' ' else vocabulary[char] for char in sentence]
        return tokens

    @staticmethod
    def decoder(idx_to_text: dict, labels: list) -> str:
        decodes = [idx_to_text[label] for label in labels]
        return ''.join(decodes)
    