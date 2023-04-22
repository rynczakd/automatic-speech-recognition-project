# utils.py
import math
import numpy as np
from PIL import Image


# Calculate first power of 2
def first_power_of_2(n: int):
    a = int(math.log2(n))

    if np.power(2, a) == n:
        return int(n)

    return int(np.power(2, (a + 1)))


# Convert Hz to Mel scale
def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


# Convert Mel scale to Hz
def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


# Compute a Mel-filterbank
def get_filter_bank(num_filters: int = 23, nfft: int = 512, sample_rate: int = 16000,
                    lower_freq: int = 0, upper_freq: int = None) -> np.ndarray:
    # Set upper frequency (the highest band edge of mel filters)
    upper_freq = upper_freq or sample_rate / 2
    assert upper_freq <= sample_rate / 2, "Upper frequency is greater than sample rate / 2"

    # Compute points evenly spaced in mel-scale
    low_mel = hz2mel(lower_freq)
    high_mel = hz2mel(upper_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)

    # Convert points from Hz to FFT bin number
    bin_points = np.floor((nfft + 1) * mel2hz(mel_points) / sample_rate)

    # Prepare filter bank array
    filter_bank = np.zeros([num_filters, nfft // 2])

    # Compute triangular function - linear B-spline, most general from of triangular function
    for j in range(0, num_filters):
        for i in range(int(bin_points[j]), int(bin_points[j + 1])):
            filter_bank[j, i] = (i - bin_points[j]) / (bin_points[j + 1] - bin_points[j])
        for i in range(int(bin_points[j + 1]), int(bin_points[j + 2])):
            filter_bank[j, i] = (bin_points[j + 2] - i) / (bin_points[j + 2] - bin_points[j + 1])

    return filter_bank


def spec2img(spectrogram: np.ndarray) -> Image.Image:
    # Scale data to obtain values from range (0, 1)
    spectrogram = (spectrogram - np.min(spectrogram)) / \
                            (np.max(spectrogram) - np.min(spectrogram))
    # Convert np.ndarray into IMG - it allows us to reduce size of our data
    # Scale image to obtain values from range (0, 255) and change data type to uint8
    spectrogram_img = Image.fromarray((spectrogram * 255).astype(np.uint8))

    return spectrogram_img


def img2spec(spectrogram_image: Image.Image) -> np.ndarray:
    # Convert IMG to np.ndarray
    spectrogram = np.array(spectrogram_image)
    # Add third dimension for CNN
    spectrogram = np.atleast_3d(spectrogram)
    spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1])
    # Scale values to range (0, 1)
    spectrogram = spectrogram / 255
    return spectrogram
