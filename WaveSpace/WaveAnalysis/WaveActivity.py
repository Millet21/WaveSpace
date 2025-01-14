import numpy as np
import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Utils.WaveData as wd

def find_wave_activity(waveData, freqList, dataBucket="ComplexPhaseData", nBases=3, ):
    waveData.ActiveDataBucket = dataBucket
    hf.assure_consistency(waveData)
    complexPhaseData = waveData.get_data(dataBucket)
    nTrials, nChannels, nTime = complexPhaseData.shape
    # Make complex valued Phase/magnitude Timeseries per freq
    #reshape to (trial, time, channel)
    phi= np.transpose(complexPhaseData, (0, 2, 1))
    bases, fit, betas = c_TW_bases_betas(phi,nBases=nBases)
    basesBucket = wd.DataBucket(bases,"Bases","chan_trl")
    fitBucket = wd.DataBucket(fit,"Fit","trl_chan")
    betasBucket = wd.DataBucket(betas,"betas","trl_chan_beta")
    waveData.add_data_bucket(basesBucket)
    waveData.add_data_bucket(fitBucket)
    waveData.add_data_bucket(betasBucket)

def c_TW_bases_betas(phi_cts,nBases=3):#phi complex-valued phase, c cases, t times, s sensors
    phi_Cs = np.asarray(phi_cts.reshape(-1,phi_cts.shape[-1]))
    phi_cent = phi_Cs - phi_Cs.mean(0)
    COV = phi_cent.T.conj()@phi_cent
    u,s,vh = np.linalg.svd(COV)
    print(100.0*s[:nBases]/s.sum())
    bases_sb = vh[:nBases].T
    betas_Cb = phi_Cs.dot(bases_sb)
    model_Cs = np.exp(1j*np.angle(bases_sb.dot(betas_Cb.T).T))
    fit_C = (phi_Cs/model_Cs).mean(-1).real
    fit_ct = fit_C.reshape(phi_cts.shape[0],phi_cts.shape[1])
    betas_ctb = betas_Cb.reshape(phi_cts.shape[0],-1,bases_sb.shape[1])
    return bases_sb,fit_ct,betas_ctb