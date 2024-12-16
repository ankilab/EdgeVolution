def prepare_stft_and_fb_genes(gene, chromosome, model, sample_rate):
    if gene['layer'] == 'STFT_2D' and len(model.layers) > 0:
            # Need to check if input_shape is smaller than n_fft, since one output dimension would be 0 then
            # If smaller, use input_shape as n_fft

            previous_layer = model.layers[-1]
            input_shape = previous_layer.input_shape  # shape: (None, shape[0], shape[1])

            if input_shape[1] < gene['n_fft']:
                gene['n_fft'] = input_shape[1]
    elif gene['layer'] == 'FB_2D': 
            # need an extra parameters when applying filterbank

            # sample rate from config.yaml
            gene['sample_rate'] = sample_rate

            # find STFT layer and get its n_fft parameter as it is needed for the filterbank layer
            stft_layer = [x for x in chromosome if x['layer'] == 'STFT_2D'][0]
            gene['n_fft'] = stft_layer['n_fft']
            
    return gene