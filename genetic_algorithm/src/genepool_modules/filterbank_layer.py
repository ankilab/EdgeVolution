from kapre import ApplyFilterbank


def get_apply_filterbank(type, sample_rate, n_fft, n_mels, mel_f_min, mel_f_max, mel_htk, mel_norm, output_data_format):
    kwargs = {
        'sample_rate': sample_rate,
        'n_freq': n_fft // 2 + 1,
        'n_mels': n_mels,
        'f_min': mel_f_min,
        'f_max': mel_f_max,
        'htk': mel_htk,
        'norm': mel_norm,
    }
    return ApplyFilterbank(type=type, filterbank_kwargs=kwargs, data_format=output_data_format)