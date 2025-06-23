    for freqInd in range(2):
        if freqInd == 0:
            freq = 5 #theta[0] we know the freq of interest here
        else:
            freq = alpha[0]
            #% do filter + Hilbert to get complex Timeseries 
        filt.filter_narrowband(waveData, dataBucketName = "EEGLayout", LowCutOff=freq-1, HighCutOff=freq+1, type = "FIR", order=100, causal=False)
        waveData.DataBuckets[str(freqInd)] =  waveData.DataBuckets.pop("NBFiltered")
    temp = np.stack((waveData.DataBuckets["0"].get_data(), waveData.DataBuckets["1"].get_data()),axis=0)
    waveData.add_data_bucket(wd.DataBucket(temp, "NBFiltered", "freq_trl_chan_time", waveData.get_channel_names()))

    # get complex timeseries
    hilb.apply_hilbert(waveData, dataBucketName = "NBFiltered")