# Automatic Speech Recognition using LSTM/GRU Neural Network and CTC Loss Function
This repository contains a future project related to the development of an ASR system using the LSTM/GRU Neural Network architecture and CTC Loss Function.

## Analysis of speech signal
**Speech signal characteristics**  
Speech is the most basic way for people to communicate. It is a sequence of complex sounds that are produced in the articulatory organs of humans. The phonetic structure of speech consists of sounds, syllables, words, phrases and sentences. The smallest part of any language is the phone, the sound of which depends not only on the way it is pronounced, but also on its placement in the word, accentuation, or the phonetic character of neighboring sounds.  

Speech as a method of human communication is characterized by specific requirements regarding the physical parameters of the signal, which can be considered in many different ways. The method of representing the signal is determined mainly by the purpose of further use - other information encoded in the speech signal is used
for its semantic recognition, and still other information for determining personal characteristics in voice biometrics. It is therefore necessary to choose an appropriate method of signal analysis.  

**Speech signal in the time domain**  
**Speech signal preprocessing**  
Preprocessing refers to a set of operations performed on a digitally recorded signal prior to its parameterization. During the preprocessing stage, a speech signal can be transformed into a set of features, the values of which can serve as a basis for describing object states in terms of speech recognition.  
Typical preprocessing operations include the _removing mean value from the speech signal_, _normalization of the signal_ and its _resampling_.  

_**Removing mean**_  
It is assumed that the mean value of the natural speech signal in the absence of interference is equal to zero. However, due to the imperfections of the acquisition process and the finite recording length, the recorded speech signals, and especially their short fragments, may contain a mean value different from zero.
It is therefore necessary to remove it by subtraction, according to the following equation:
$$x^{(m)}(n) = x(n) - m(n)$$  where 𝑥(𝑛) denotes the recorded speech signal and 𝑚(𝑛) represents the mean value.

Determining the exact mean value is impossible because it would require it to have an infinite-time signal. The estimator of the mean value is therefore determined from the recorded portion of the speech signal under the assumption that it is a function of time in the general case. If the length of the recorded
speech is equal to 𝑁, the time-independent estimator 𝑚 is determined according to the according to the relation:
$$m = \frac{1} {N} \sum_{n=0}^{N-1} x(n)$$




**Speech signal in the time-frequency domain**  
**Mel fiters**  
**Log-mel spectrograms**  
