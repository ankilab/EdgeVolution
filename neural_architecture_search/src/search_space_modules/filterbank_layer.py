from kapre import ApplyFilterbank

from neural_architecture_search.src.layer_registry import LayerRegistry


@LayerRegistry.register(metadata={"source": "custom", "category": "preprocessing"})
def get_filterbank_layer(
    type, sample_rate, n_fft, n_mels, mel_f_min, mel_f_max, output_data_format
):
    """
    Returns a filterbank layer for audio feature extraction.

    Args:
        type: Type of filterbank ('mel', etc.)
        sample_rate: Audio sample rate in Hz
        n_fft: FFT size
        n_mels: Number of mel bands
        mel_f_min: Minimum frequency for mel scale
        mel_f_max: Maximum frequency for mel scale
        output_data_format: Data format for output ('channels_last' or 'channels_first')

    Returns:
        ApplyFilterbank layer configured with the given parameters
    """
    kwargs = {
        "sample_rate": sample_rate,
        "n_freq": n_fft // 2 + 1,
        "n_mels": n_mels,
        "f_min": mel_f_min,
        "f_max": mel_f_max,
        "htk": False,
        "norm": "slaney",
    }
    return ApplyFilterbank(
        type=type, filterbank_kwargs=kwargs, data_format=output_data_format
    )
