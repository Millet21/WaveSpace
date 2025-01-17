import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Utils.WaveData as wd
import numpy as np
from scipy.signal import morlet2

def freq_domain_wavelet(waveData, dataBucket, freqlist): 
    """computes the wavelet transform of input data using 
    the computationally efficient Morlet wavelet in the frequency domain. 
    The wavelet transform is computed using the Fast Fourier Transform (FFT) for each trial, channel, and frequency. 
    Well-suited for signals with a broad range of frequencies, but, due to the nature of the FFT, the wavelets are 
    implicitly assumed to be infinite in length, which can lead to some loss of accuracy, particularly at lower frequencies. 
    For higher accuracy use time-domain convolution approach instead"""
    waveData.ActiveDataBucket= dataBucket
    hf.assure_consistency(waveData)
    data = waveData.get_data(dataBucket)    
    sfreq = waveData.get_sample_rate()
    # Compute the TFR using the Morlet wavelet transform
    n_cycles = 5
    tfr = tfr_morlet(data, freqlist, n_cycles)
    ComplexPhaseDataBucket = wd.DataBucket(tfr, "ComplexPhaseData", "trl_freq_chan_time", waveData.get_channel_names())
    waveData.add_data_bucket(ComplexPhaseDataBucket)


def tfr_morlet(data, freqs, n_cycles):
    n_trials, n_channels, n_times = data.shape
    n_frequencies = len(freqs)
    tfr = np.zeros(( n_trials, n_frequencies, n_channels,  n_times), dtype=np.complex128)
    for i_trial in range(n_trials):
        for i_ch in range(n_channels):
            for i_freq, freq in enumerate(freqs):
                w = morlet2(n_times, n_cycles / (freq * 2 * np.pi), w=5)
                data_tf = np.fft.fft(data[i_trial, i_ch, :] * w)
                tfr[i_trial,i_freq, i_ch, : ] = data_tf
    return tfr

#################
import numpy as np
from scipy.optimize import curve_fit

def convolution_wavelet(waveData,dataBucket, frequencies, N_cycles=2):
    """Returns the wavelet transformed data via convolution in the time domain. This approach is computationally more expensive 
    than the freq_domain_wavelet, but can improve accuracy, especially at lower frequencies, by explicitly accounting for the finite length of the wavelets.
    N_cycles is the number of cycles in the Morlet wavelet; typically 2 in Alexander et al.
    Frequencies are the center frequencies, typically chosen from oversampled, logarithmically spaced frequencies
    minf is the minimum frequency used in the analysis, needed to ensure the final phase/power estimates all have the same number of samples
    https://doi.org/10.1371/journal.pone.0148413
    https://doi.org/10.1371/journal.pcbi.1007316"""
    waveData.ActiveDataBucket = dataBucket
    hf.assure_consistency(waveData)
    data = waveData.get_data(dataBucket)
    deltaT = 1000/waveData.get_sample_rate()
    minf = np.min(frequencies)
    nTrials, nSensors, nTimes = data.shape
    PH_unpad_at_minf = int(N_cycles*500.0/(minf*deltaT))
    #the number of samples of phase/power that will be generated
    PHSamples = nTimes - 2*PH_unpad_at_minf
    #check if data segment is long enough
    if PHSamples < 1:
        raise ValueError('Data segment too short for selected frequencies and N_cycles')
    #half the wavelet cycles at lowest frequency minus half the wavelet cycles at this frequency
    power = np.zeros((nTrials,len(frequencies),nSensors,PHSamples),float)
    phi = np.zeros((nTrials,len(frequencies),nSensors,PHSamples),complex)
    for freqInd, frequency in enumerate(frequencies):
        PH_pad_at_f =  PH_unpad_at_minf - int(N_cycles*500.0/(frequency*deltaT))
        N_cycles_of_samples = int(N_cycles*1000.0/(frequency*deltaT))
        #the extra samples of data required at both start and beginning of estimated phase/power values
        W = gaussian_window(N_cycles_of_samples) #size of Morlet window
        f_indices = []
        #how many sets of indexes to make? one for each sample in phase
        for t in range(PHSamples):
            #make a list of all data samples that are used in calculating each phase estimate
            padded_range = range(t+PH_pad_at_f,t+PH_pad_at_f+N_cycles_of_samples)
            f_indices += padded_range
        f_indices = np.asarray(f_indices)
        for trl,thistrial in enumerate(data):
            for channel in range(nSensors):
                data_tw = thistrial[channel,f_indices].reshape(PHSamples,N_cycles_of_samples) #tw
                base = data_tw.mean(1) #over w
                data_tw = data_tw - base[:,np.newaxis]
                t_f = 2.0*np.pi*N_cycles*np.arange(N_cycles_of_samples)/N_cycles_of_samples
                t_phi = np.exp(1j*(t_f)) * W
                fourier_components = (data_tw * t_phi[np.newaxis,:]).sum(1)
                phi[trl,freqInd,channel,:] = np.exp(1j*np.angle(fourier_components)) #over w
                power[trl,freqInd,channel,:] =  np.absolute(fourier_components)

    ComplexPhaseDataBucket = wd.DataBucket(phi, "ComplexPhaseData", "trl_freq_chan_time",waveData.get_channel_names())
    PowerDataBucket = wd.DataBucket(power, "Power", "trl_freq_chan_time",waveData.get_channel_names())
    waveData.add_data_bucket(ComplexPhaseDataBucket)
    waveData.add_data_bucket(PowerDataBucket)


def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gaussian_window(x):
    if type(x) is int:
        n = x
        x = np.arange(n)
    else:
        n = len(x)
    cosine_window = 0.5*(1.0 - np.cos(2*np.pi*x/(n-1)))
    mean = sum(x*cosine_window)/n
    sigma = np.sqrt(sum(cosine_window*(x-mean)**2)/n)
    popt,pcov = curve_fit(gauss,x,cosine_window,p0=[1,mean,sigma])
    G = gauss(x,*popt)
    return G