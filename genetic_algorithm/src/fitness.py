

def calculate_fitness(acc, mem, inf, enc, params):
    a = params['acc_weight']
    b = params['mem_weight']
    c = params['inf_weight']
    d = params['enc_weight']

    return a * acc + b * mem + c * inf + d * enc
