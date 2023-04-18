# Automatic Speech Recognition using LSTM/GRU Neural Network and CTC Loss Function
This repository contains a future project related to the development of an ASR system using the LSTM/GRU Neural Network architecture and CTC Loss Function.

## Analysis of speech signal
**Speech signal characteristics**  
Speech is the most basic way for people to communicate. It is a sequence of complex sounds that are produced in the articulatory organs of humans. The phonetic structure of speech consists of sounds, syllables, words, phrases and sentences. The smallest part of any language is the phone, the sound of which depends not only on the way it is pronounced, but also on its placement in the word, accentuation, or the phonetic character of neighboring sounds.  

Speech as a method of human communication is characterized by specific requirements regarding the physical parameters of the signal, which can be considered in many different ways. The method of representing the signal is determined mainly by the purpose of further use - other information encoded in the speech signal is used
for its semantic recognition, and still other information for determining personal characteristics in voice biometrics systems. It is therefore necessary to choose an appropriate method of signal analysis. For processing speech based on deep learning, the log-magnitude mel-spectrogram has been widely used as the feature representation. The rest of that section will describe the full process of generating such spectrograms.

**Speech signal in the time domain**  
Speech signal by its nature is categorized as stochastic, non-stationary signal. _Stochastic_ refers to the fact that the model of this signal is a real
stochastic process described by random variables with known statistics. _Non-stationary_ means that second-order statistics of the speech signal are time-varying.  
However, there is a concept of local stationarity, based on which, over sufficiently short time intervals, a speech signal can be treated as a stationary process.  

In the time domain, speech signal is represented as a sequence of amplitude values over time. These amplitude values represent the variations in air pressure created by the movement of the speaker's vocal tract and are commonly referred to as the speech waveform. In simple terms, a speech signal in the time domain is considered as a long vector containing real values:
$$\mathbf{x} \in \mathbb{R}^N$$  
where _N_ denotes a total signal length (total number of samples) and can be calculated as _signal duration_ (in seconds) times _sampling frequency_ (number of samples per second).  

One common representation of speech in the time domain is the waveform plot, which displays the amplitude values of the speech signal as a function of time. An example waveform plot for the sentence is shown below.

<img src="https://github.com/rynczakd/automatic-speech-recognition-project/blob/main/images/waveform_plot.png" width="600">

**Speech signal preprocessing**  
Preprocessing refers to a set of operations performed on a recorded signal prior to its parameterization. During the preprocessing stage, a speech signal can be transformed into a set of features, the values of which can serve as a basis for describing object states in terms of speech recognition.  
Typical preprocessing operations include the _removing mean_ value from the speech signal and _normalization_ of the signal.

_**Removing mean**_  
It is assumed that the mean value of the natural speech signal in the absence of interference is equal to zero. However, due to the imperfections of the acquisition process and the finite recording length, the recorded speech signals, and especially their short fragments, may contain a mean value different from zero.
It is therefore necessary to remove it by subtraction, according to the following equation:
$$x^{(m)}(n) = x(n) - m(n)$$  where 𝑥(𝑛) denotes the recorded speech signal and 𝑚(𝑛) represents the mean value.

Determining the exact mean value is impossible because it would require it to have an infinite-time signal. The estimator of the mean value is therefore determined from the recorded portion of the speech signal under the assumption that it is a function of time in the general case. If the length of the recorded
speech is equal to 𝑁, the time-independent estimator 𝑚 is determined according to the relation:
$$m = \frac{1} {N} \sum_{n=0}^{N-1} x(n)$$

_**Normalization**_  
The normalization operation involves dividing the signal samples by the appropriate value. In practice, the following parameters are used for normalization: _variance_, _mean absolute value_, _maximum momentary value_. In the case of the project, the last parameter was chosen.  

After removing the mean value, the maximum momentary value 𝐴𝑗 for the j-th utterance fragment of length 𝑁 can be expressed as follows:
$$A_j^{\max} = \max\limits_{N} \left| x(n) \right|$$  

Signal normalization has the effect of preserving the energy relationships between individual phonemes of an utterance. This step also involves scaling the audio signal so that it falls within a certain range, such as -1 to 1 or 0 to 1. This can help to prevent clipping or distortion in the audio signal.

<img src="https://github.com/rynczakd/automatic-speech-recognition-project/blob/main/images/waveforms_comparison.png" width="700">


**Speech signal in the time-frequency domain**  
Analysis of non-stationary signals, including the acoustic signal of speech, involves determining the changes in the values of harmonic amplitudes occurring over time. Observation of variation of the spectrum with respect to time is made possible by applying the Short-Time Fourier Transform (STFT).

The STFT is a commonly used signal processing technique for analyzing non-stationary signals. It provides a way to analyze the frequency content of a signal over time by computing the Fourier transform of small, overlapping sections (frames) of the signal. This allows us to examine the spectral characteristics of a signal as they evolve over time.

To apply the STFT to a speech signal vector **x**, we first need to split that signal into a series of overlapping frames. It is assumed that for a speech signal the length of the analysis window should be 25 ms - the signal is then considered locally stationary, which is the basis for further analysis. The frames are usually shifted by a fixed amount of samples (e.g. half the frame size) to ensure that they overlap. For speech analysis it is assumed that overlap should be 10 ms.  
By arranging each of the extracted frames side by side, we can form a speech signal matrix _X_: 
$$X \in \mathbb{R}^{k \times l}$$  
where _k_ denotes the length of the signal frame (in samples), while _l_ corresponds to the total number of signal frames that we can obtain from the speech signal vector (with overlap).

Then, to reduce spectral leakage and improve the frequency resolution of the STFT, we typically apply a window function to each frame.  
A commonly used window function is the Hamming window which looks as follows:  
$$w(n) = 0.54 - 0.46 \cos\left(\frac{2 \pi n}{N-1}\right)$$  
where _N_ corresponds to the number of samples in the signal frame.

Once we have split the signal into overlapping frames, built a signal matrix _X_ and applied a window function to each frame (column of the matrix), we can compute the Fourier transform of each frame. The Fourier transform of a frame gives us a measure of the frequency content of the signal within that frame.  
The Fourier transform is defined by the following formula:  
$$\widetilde{X}[k] = \sum_{n=0}^{N-1} x[n]w[n] e^{-j2\pi kn/N},   k = 0, 1, ..., K$$  
where _K_ denotes the number of discrete Fourier transform coefficients (FFT size). The size of that parameter _K_ is typically chosen to be a power of 2 to allow for efficient computation of the Fourier transform. The resulting FFT spectrum is symmetric, and we can take only half of it to reconstruct the signal.  
The first FFT coefficient corresponds to the DC component of the signal, and each subsequent coefficient corresponds to a frequency that is a multiple of frequency resolution. The last coefficient corresponds to the Nyquist frequency, which is half of the sampling rate. The frequency resolution of the FFT is given by the sampling rate divided by the FFT size:
$$\Delta f = \frac{f_s}{N_{FFT}}$$  

Note that, it is not necessary for the frame size to be equal to the FFT size. However, it is common practice to choose the frame size such that is a multiple of the FFT size or a power of 2, to simplify the FFT computation and improve computational efficiency. In the case where the frame size is less than the FFT size it is recommended to zero pad each frame before computing its FFT to achieve a power-of-two FFT size. Zero padding involves adding zeros to the end of the frame until it reaches the desired length. Zero padding should be applied after the window function is applied to the signal. The reason for this is that applying a window function to a frame of speech signal reduces the amount of spectral leakage in the resulting FFT, which can cause unwanted artifacts in the frequency spectrum. If we apply zero padding before applying the window function, the additional zeros in the padded region will not be affected by the window function and will contribute to spectral leakage in the FFT. 

Finally, we concatenate the Fourier transforms of each frame to obtain the STFT of the entire audio signal. The resulting STFT is a 2D matrix X̂ where each row represents a frequency band and each column represents a time window. Fourier transform is going to take us from real numbers to complex numbers:
$$X \rightarrow \widetilde{X} \in \mathbb{C}^{K \times l}$$  
where _K_ denotes number of FFT coefficient for single frame (FFT size) and _l_ corresponds to the total number of signal frames. A visual representation of STFT is called spectrogram. A typical format for a spectrogram is a two-dimensional heat map that shows the frequency content of a signal over time. The horizontal axis represents time, while the vertical axis represents frequency. The amplitude or power of each frequency component at a given point in time is represented by the intensity or color of each point in the image.  

**Log-magnitude spectrograms**  
In speech processing, the complex-valued spectrogram is often used to represent the frequency content of a signal over time. However, in many audio applications, the phase information is not as important as the magnitude information (this does not mean that phase information has no use, as there are publications that talk about using that information as a part of data for training deep learning models).  
In order to perform spectral magnitude estimation, we can simply take the absolute value of each element in the STFT matrix X̂: 
$$MS = |\widetilde{X}|, \qquad MS \in \mathbb{R}^{K \times l}$$
where as before _K_ denotes FFT size and _l_ corresponds to the total number of signal frames.  

We can also perform Periodogram estimate of the power spectrum according to the following equation:  
$$P = \frac{1}{N}|\widetilde{X}|^2, \qquad P \in \mathbb{R}^{K \times l}$$
where _N_ denotes the number of samples in single frame.  

Then, in order to compress the dynamic range of the spectrogram and to make it more suitable for visualizing and processing we can take the logarithm of the magnitude spectrogram:  
$$logMS = \log_{10}(MS), \qquad MS \in \mathbb{R}^{K \times l}$$

<img src="https://github.com/rynczakd/automatic-speech-recognition-project/blob/main/images/log-magnitude-spectrogram.png" width="1200">

**Mel fiters**  
The human perception of sound is not linearly related to the physical properties of sound waves such as frequency. Instead, our perception is more closely related to the perceived pitch of sound, which is a nonlinear function of frequency. The Mel scale is a perceptual scale that approximates the nonlinear relationship between frequency and pitch. By transforming the frequency axis of a spectrogram to the Mel scale, we can create a new representation that is more closely related to the way humans perceive sound.  

To convert a log-magnitude spectrogram to log-magnitude mel-spectrogram, we need to apply a mel filterbank to the spectrogram. The mel filterbank essentially groups adjacent frequency bins in the spectrogram into a set of mel-frequency bins, which are spaced according to the human auditory system's perception of pitch. The mel filterbank is a set of triangular bandpass filters that are spaced evenly in the mel-frequency domain. The output of each filter is the weighted sum of the power spectrogram values in the corresponding frequency bin.  

To create mel filterbank, we need to first define the parameters of the filterbank:  
- _lower frequency_ - minimum signal frequency, in the case of speech signal analysis lower frequency is usually equal to 0 Hz,
- _upper frequency_ - maximum signal frequency, in the case of speech signal analysis upper frequency is usually equal to (_sample rate_/2) Hz,
- _number of filters_ - number of mel-frequency bins in the filterbank.  

Then, we can compute the center frequencies of the mel-frequency bins. To convert frequencies in terms of Hertz to mel-scale we can use the following formula:  
$$M(f) = 2595 \log_{10}(1 + \frac{f}{700.0})$$  
Knowing the above equation, we can calculate the minimum and maximum mel frequencies and then divide the mel frequency range into equally spaced points:
$$M(\operatorname{linspace}(M(\mathrm{lower\ frequency})\, M(\mathrm{upper\ frequency})\, \mathrm{number\ of\ filters}+2))$$  
The reason for adding 2 to the number of filters in the equation above is to include the lower and upper frequency bounds in the filterbank.

Then, we can convert each mel frequency point back to Hertz using the inverse mel frequency formula. To convert mel-scale values to Hertz we can use following formula:
$$M^{-1}(m) = 700(10 ^ {\frac{m}{2595}} - 1)$$  
Our inverse mel frequency formula looks therefore as follows:
$$M{-1}(\operatorname{linspace}(M(\mathrm{lower\ frequency})\, M(\mathrm{upper\ frequency})\, \mathrm{number\ of\ filters}+2))$$  

Note that in order to convert the obtained frequencies into FFT bins, we need to multiply the obtained frequency values by the scaling factor: 
$$\frac{nfft + 1}{sample rate}$$
where the _(nfft + 1)_ in the numerator is added beacuse the FFT coefficient are symmetric, and the +1 ensures that the highest frequency bin is included. Then we have to round down resulting Hertz values to the nearest FFT bin index. The scaling factor is derived from the frequency resolution of the FFT mentioned earlier.   This step is necessary because the filter bank needs to be applied to the power spectrogram, which is computed using the FFT, so the mel frequency points need to be converted to FFT bin numbers.  

Finally, we can compute the filterbank weights for each frequency bin. For this purpose, it is necessary to create empty filterbank matrix with (_number of filters_ x _nfft_/2) and then compute the filter weigths for each frequency bin. The filter weights are computed using a triangular function that is a piecewise linear function that varies from 0 at the left frequency to 1 at the center frequency and back to 0 at the right frequency (in its most general form a triangular function is any linear B-spline).  

The filter weights are stored in the filterbank matrix _T_, where each row corresponds to one filter:
$$T \in \mathbb{R}^{t \times n}$$  
where _t_ denotes number of filters and _n_ corresponds to (_nfft_/2) - the filter bank matrix is typically only computed up to the nfft/2 frequency component. We can also normalize the filters by dividing each row by its sum to ensure that the sum of the weights is 1.

**Log-magnitude mel spectrograms**  
To convert our log-magnitude spectrogram to a log-magnitude mel spectrogram, we need to multiply the filterbank matrix _T_ by a log-magnitude spectrogram matrix_MS_. The resulting matrix will be a log-magnitude mel-spectrogram _E_:
$$E = T \cdot MS , \qquad E \in \mathbb{R}^{t \times n}$$  
where _t_ denotes number of filters and _n_ corresponds to (_nfft_/2). The values in a log-magnitude mel spectrogram represent the amount of energy of the audio signal at different frequency bands over time.

<img src="https://github.com/rynczakd/automatic-speech-recognition-project/blob/main/images/log-magnitude-mel-spectrogram.png" width="1200">
<img src="https://github.com/rynczakd/automatic-speech-recognition-project/blob/main/images/waveform-with-log-magnitude-mel-spec.png" width="600">


**_Note that for all calculations, the spectrogram matrix frequencies are limited to nfft/2_.**
  
