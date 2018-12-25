from keras import backend as K


# def huber_loss(y, q_value):
#     error = K.abs(y - q_value)
#     quadratic_part = K.clip(error, 0.0, 1.0)
#     linear_part = error - quadratic_part
#     loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
#     return loss

def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error * error / 2
    linear_term = abs(error) - 1 / 2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term