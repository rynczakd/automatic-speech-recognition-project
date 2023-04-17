# Automatic Speech Recognition using LSTM/GRU Neural Network and CTC Loss Function
This repository contains a future project related to the development of an ASR system using the LSTM/GRU Neural Network architecture and CTC Loss Function.

## Analysis of speech signal
**Speech signal characteristics**  
Speech is the most basic way for people to communicate. It is a sequence of complex sounds that are produced in the articulatory organs of humans. The phonetic structure of speech consists of sounds, syllables, words, phrases and sentences. The smallest part of any language is the phone, the sound of which depends not only on the way it is pronounced, but also on its placement in the word, accentuation, or the phonetic character of neighboring sounds.  

Speech as a method of human communication is characterized by specific requirements regarding the physical parameters of the signal, which can be considered in many different ways. The method of representing the signal is determined mainly by the purpose of further use - other information encoded in the speech signal is used
for its semantic recognition, and still other information for determining personal characteristics in voice biometrics. It is therefore necessary to choose an appropriate method of signal analysis.  

**Speech signal in the time domain**  
Speech signal by its nature is categorized as stochastic, non-stationary signal. _Stochastic_ refers to the fact that the model of this signal is a real
stochastic process described by random variables with known statistics. _Non-stationary_ means that second-order statistics of the speech signal are time-varying.  
However, there is a concept of local stationarity, based on which, over sufficiently short time intervals, a speech signal can be treated as a stationary process.  

In the time domain, speech signal is represented as a sequence of amplitude values over time. These amplitude values represent the variations in air pressure created by the movement of the speaker's vocal tract and are commonly referred to as the speech waveform. In simple terms, a speech signal in the time domain is considered as a long vector containing real values:
$$\mathbf{x} \in \mathbb{R}^n$$

where _n_ can be calculated as _number of samples_ times _sampling frequency_ (number of samples per second).  

One common representation of speech in the time domain is the waveform plot, which displays the amplitude values of the speech signal as a function of time. An example waveform plot for English statements is shown below.  

**Speech signal preprocessing**  
Preprocessing refers to a set of operations performed on a digitally recorded signal prior to its parameterization. During the preprocessing stage, a speech signal can be transformed into a set of features, the values of which can serve as a basis for describing object states in terms of speech recognition.  
Typical preprocessing operations include the _removing mean value from the speech signal_ and _normalization of the signal_.

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


**Speech signal in the time-frequency domain**  
Analysis of non-stationary signals, including the acoustic signal of speech, involves determining the changes in the values of harmonic amplitudes occurring over time. Observation of variation of the spectrum with respect to time is made possible by applying the Short-Time Fourier Transform (STFT).

The STFT is a commonly used signal processing technique for analyzing non-stationary signals. It provides a way to analyze the frequency content of a signal over time by computing the Fourier transform of small, overlapping sections (frames) of the signal. This allows us to examine the spectral characteristics of a signal as they evolve over time.

To apply the STFT to a speech signal vector **x**, we first need to split that signal into a series of overlapping frames. It is assumed that for a speech signal the length of the analysis window should be 25 ms. The frames are usually shifted by a fixed amount (e.g. half the frame size) to ensure that they overlap. For speech analysis it is assumed that overlap should be 10 ms.  
By arranging each of the extracted frames side by side, we can form a speech signal matrix _X_: 
$$X \in \mathbb{R}^{k \times l}$$  
where _k_ index is the length of the signal frame, while _l_ corresponds to the total number of signal frames that we can obtain from the speech signal vector (with overlap).

Then, to reduce spectral leakage and improve the frequency resolution of the STFT, we typically apply a window function to each frame.  
A commonly used window function is the Hamming window which looks as follows:  
$$w(n) = 0.54 - 0.46 \cos\left(\frac{2 \pi n}{N-1}\right)$$  
where _N_ corresponds to the number of samples in the signal frame.

Once we have split the signal into overlapping frames, built a signal matrix _X_ and applied a window function to each frame (column of the matrix), we can compute the Fourier transform of each frame. The Fourier transform of a frame gives us a measure of the frequency content of the signal within that frame.  
The Fourier transform is defined by the following formula:  
$$\widetilde{X}[k] = \sum_{n=0}^{N-1} x[n]w[n] e^{-j2\pi kn/N},   k = 0, 1, ..., K$$  
where _K_ denotes the number of discrete Fourier transform coefficients. The size of that parameter _K_ is typically chosen to be a power of 2 to allow for efficient computation of the Fourier transform.  

Finally, we concatenate the Fourier transforms of each frame to obtain the STFT of the entire audio signal. The resulting STFT is a 2D matrix X̂ where each row represents a frequency band and each column represents a time window. Fourier transform is going to take us from real numbers to complex numbers:
$$X \rightarrow \widetilde{X} \in \mathbb{C}^{K \times l}$$  
where _K_ denotes the FFT size and _l_ corresponds to the total number of signal frames

**Mel fiters**  
**Log-mel spectrograms**  
