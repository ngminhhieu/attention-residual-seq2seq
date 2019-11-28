# Stacked LSTM with residual connections in depth direction.
#
# Naturally LSTM has something like residual connections in time.
# Here we add residual connection in depth.
#
# Inspired by Google's Neural Machine Translation System (https://arxiv.org/abs/1609.08144).
# They observed that residual connections allow them to use much deeper stacked RNNs.
# Without residual connections they were limited to around 4 layers of depth.
#
# It uses Keras 2 API.

from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers.merge import add


# def Residual_enc(input, rnn_unit, rnn_depth, rnn_dropout):
#     """
#     The intermediate LSTM layers return sequences, while the last returns a single element.
#     The input is also a sequence. In order to match the shape of input and output of the LSTM
#     to sum them we can do it only for all layers but the last.
#     """

#     x = input
#     for i in range(rnn_depth):
#         x_rnn, state_h, state_c = LSTM(rnn_unit, return_sequences=True,
#                                    return_state=True, name='LSTM_enc_{}'.format(i+1))(x)
#         # Intermediate layers return sequences, input is also a sequence.
#         if i > 0:
#             x = add([x, x_rnn])
#         else:
#             # Note that the input size and RNN output has to match, due to the sum operation.
#             # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
#             x = x_rnn

#     return x, state_h, state_c


# def Residual_dec(input, rnn_unit, rnn_depth, rnn_dropout, init_states):
#     """
#     The intermediate LSTM layers return sequences, while the last returns a single element.
#     The input is also a sequence. In order to match the shape of input and output of the LSTM
#     to sum them we can do it only for all layers but the last.
#     """
#     layers = []
#     x = input
#     layer = LSTM(rnn_unit, return_sequences=True,
#                                    return_state=True, name='LSTM_dec_{}'.format(1))
#     x_rnn, state_h, state_c = layer(input, initial_state=init_states)
#     x = x_rnn
#     layers.append(layer)

#     for i in range(1, rnn_depth, 1):
#     # for i in range(1,rnn_depth):
#         layer = LSTM(rnn_unit, return_sequences=True,
#                                        return_state=True, name='LSTM_dec_{}'.format(i+1))
#         layers.append(layer)                                       
#         x_rnn, state_h, state_c = layer(x)
#         # Intermediate layers return sequences, input is also a sequence.
#         if i > 0:
#             x = add([x, x_rnn])
#         else:
#             # Note that the input size and RNN output has to match, due to the sum operation.
#             # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
#             x = x_rnn

#     return layers, x, state_h, state_c

def Residual_enc(input, rnn_unit, rnn_depth, rnn_dropout):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """

    x = input
    state_h, state_c = None, None
    for i in range(rnn_depth):
        x_rnn, state_h, state_c = LSTM(rnn_unit, return_sequences=True,
                                   return_state=True, name='LSTM_enc_{}'.format(i))(x)

        # Intermediate layers return sequences, input is also a sequence.
        if i > 0:
            x = add([x, x_rnn])
        else:
            # Note that the input size and RNN output has to match, due to the sum operation.
            # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
            x = x_rnn

    return x, state_h, state_c


def Residual_dec(input, rnn_unit, rnn_depth, rnn_dropout, init_states):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """

    x = input

    decoder = LSTM(rnn_unit, return_sequences=True,
                                   return_state=True, name='LSTM_dec_{}'.format(0))
    x_rnn, state_h, state_c = decoder(x, initial_state=init_states)
    x = x_rnn

    for i in range(1, rnn_depth, 1):
        x_rnn, state_h, state_c = LSTM(rnn_unit, return_sequences=True,
                                       return_state=True, name='LSTM_dec_{}'.format(i))(x)
        # Intermediate layers return sequences, input is also a sequence.
        if i > 0:
            x = add([x, x_rnn])
        else:
            # Note that the input size and RNN output has to match, due to the sum operation.
            # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
            x = x_rnn

    return x, state_h, state_c
