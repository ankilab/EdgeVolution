from kapre import ApplyFilterbank


def get_filterbank_layer(type, sample_rate, n_fft, n_mels, mel_f_min, mel_f_max, output_data_format):
    kwargs = {
        'sample_rate': sample_rate,
        'n_freq': n_fft // 2 + 1,
        'n_mels': n_mels,
        'f_min': mel_f_min,
        'f_max': mel_f_max,
        'htk': False,
        'norm': 'slaney',
    }
    return ApplyFilterbank(type=type, filterbank_kwargs=kwargs, data_format=output_data_format)