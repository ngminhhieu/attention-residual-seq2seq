import tensorflow as tf
import os
import time
import keras.callbacks as keras_callbacks
import numpy as np
import pandas as pd
import yaml
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Input
from keras.models import Model
from keras.utils import plot_model
from tqdm import tqdm
from lib import utils
from model.residual_lstm import ResidualLSTM
from model.attention import BahdanauAttention, Attention_decoder


class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class AttentionResidualSeq2SeqSupervisor():

    def __init__(self, is_training=True, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        self._test_kwargs = kwargs.get('test')
        self._model_kwargs = kwargs.get('model')
        self._alg_name = self._kwargs.get('alg')

        # data args
        self._dataset = self._data_kwargs.get('dataset')
        self._test_size = self._data_kwargs.get('test_size')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._logger.info(kwargs)

        # Model's Args
        self._model_type = self._model_kwargs.get('model_type')
        self._verified_percentage = self._model_kwargs.get('verified_percentage')
        self._rnn_units = self._model_kwargs.get('rnn_units')
        self._seq_len = self._model_kwargs.get('seq_len')
        self._horizon = self._model_kwargs.get('horizon')
        self._input_dim = self._model_kwargs.get('input_dim')
        self._output_dim = self._model_kwargs.get('output_dim')
        self._nodes = self._model_kwargs.get('num_nodes')
        self._rnn_layers = self._model_kwargs.get('rnn_layers')

        # Train's args
        self._drop_out = self._train_kwargs.get('dropout')
        self._epochs = self._train_kwargs.get('epochs')
        self._batch_size = self._data_kwargs.get('batch_size')
        self._optimizer = self._train_kwargs.get('optimizer')

        # Test's args
        self._run_times = self._test_kwargs.get('run_times')

        # Load data
        self._data = utils.load_dataset_lstm_ed(seq_len=self._seq_len, horizon=self._horizon,
                                                    input_dim=self._input_dim, output_dim=self._output_dim,
                                                    dataset=self._dataset,
                                                    r=self._verified_percentage, p=self._test_size, **kwargs)
        self.callbacks_list = []

        self._checkpoints = ModelCheckpoint( self._log_dir + "best_model.hdf5",
                                            monitor='val_loss', verbose=1, save_best_only=True,mode='auto', period=1)
        self._earlystop = EarlyStopping(monitor='val_loss', patience=self._train_kwargs.get('patience'),
                                        verbose=1, mode='auto')
        self._time_callback = TimeHistory()
        
        self.callbacks_list.append(self._checkpoints)
        self.callbacks_list.append(self._earlystop)
        self.callbacks_list.append(self._time_callback)

        self.model = self._model_construction(is_training=is_training)

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            rnn_layers = kwargs['model'].get('rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(rnn_layers)])
            seq_len = kwargs['model'].get('seq_len')
            horizon = kwargs['model'].get('horizon')

            model_type = kwargs['model'].get('model_type')
            verified_percentage = kwargs['model'].get('verified_percentage')

            run_id = '%s_%d_%d_%s_%d_%g/' % (
                model_type,seq_len, horizon,
                structure, batch_size, verified_percentage)
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def _model_construction(self, is_training=True):
        # Model
        encoder_inputs = Input(shape=(None, self._input_dim))
        encoder_outputs, state_h, state_c = ResidualLSTM(encoder_inputs, rnn_unit=self._rnn_units,
                                                         rnn_depth=self._rnn_layers, rnn_dropout=self._drop_out)

        encoder_states = [state_h, state_c]

        attention_layer = BahdanauAttention(self._rnn_units)


        decoder_inputs = Input(shape=(None, self._output_dim))

        decoder_outputs, _, _ = Attention_decoder(decoder_inputs,state_h,state_c, self._rnn_units,self._rnn_layers,
                                                  self._drop_out)
        
        # dense decoder_outputs
        decoder_dense = Dense(self._output_dim, activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        if is_training:
            return model
        else:
            self._logger.info("Load model from: {}".format(self._log_dir))
            model.load_weights(self._log_dir + 'best_model.hdf5')
            model.compile(optimizer=self._optimizer, loss='mse', metrics=['mse', 'mae'])

            # Construct E_D model for predicting
            encoder_states = encoder_outputs[1:]
            self.encoder_model = Model(encoder_inputs, encoder_states)

            decoder_state_input_h = Input(shape=(self._rnn_units,))
            decoder_state_input_c = Input(shape=(self._rnn_units,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

            decoder_outputs = Attention_decoder(decoder_inputs,state_h,state_c, self._rnn_units,self._rnn_layers,
                                                  self._drop_out)
            decoder_states = decoder_outputs[1]
            decoder_outputs = decoder_dense(decoder_outputs)
            self.decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)

            plot_model(model=self.encoder_model, to_file=self._log_dir + '/encoder.png', show_shapes=True)
            plot_model(model=self.decoder_model, to_file=self._log_dir + '/decoder.png', show_shapes=True)

            return model

    def train(self):
        self.model.compile(optimizer=self._optimizer, loss='mse', metrics=['mse', 'mae'])

        training_history = self.model.fit([self._data['encoder_input_train'], self._data['decoder_input_train']],
                                          self._data['decoder_target_train'],
                                          batch_size=self._batch_size,
                                          epochs=self._epochs,
                                          callbacks=self.callbacks_list,
                                          validation_data=([self._data['encoder_input_val'],
                                                            self._data['decoder_input_val']],
                                                           self._data['decoder_target_val']),
                                          shuffle=True,
                                          verbose=2)
        if training_history is not None:
            self._plot_training_history(training_history)
            self._save_model_history(training_history)
            config = dict(self._kwargs)
            config_filename = 'config_lstm.yaml'
            config['train']['log_dir'] = self._log_dir
            with open(os.path.join(self._log_dir, config_filename), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

    def evaluate(self):
        # todo:
        pass

    def test(self):
        for time in range(self._run_times):
            print('TIME: ', time+1)
            self._test()

    def _test(self):
        scaler = self._data['scaler']
        data_test = self._data['test_data_norm']
        T = len(data_test)
        K = data_test.shape[1]
        bm = utils.binary_matrix(self._verified_percentage, len(data_test), self._nodes)
        l = self._seq_len
        h = self._horizon
        pd = np.zeros(shape=(T - h, self._nodes), dtype='float32')
        pd[:l] = data_test[:l]
        _pd = np.zeros(shape=(T - h, self._nodes), dtype='float32')
        _pd[:l] = data_test[:l]
        iterator = tqdm(range(0, T - l - h, h))
        for i in iterator:
            if i+l+h > T-h:
                # trimm all zero lines
                pd = pd[~np.all(pd==0, axis=1)]
                _pd = _pd[~np.all(_pd==0, axis=1)]
                iterator.close()
                break
            for k in range(K):
                input = np.zeros(shape=(1, l, self._input_dim))
                # input_dim = 2
                input[0, :, 0] = pd[i:i + l, k]
                # input[0, :, 1] = bm[i:i + l, k]
                yhats = self._predict(input)
                yhats = np.squeeze(yhats, axis=-1)
                _pd[i + l:i + l + h, k] = yhats
                # update y
                _bm = bm[i + l:i + l + h, k].copy()
                _gt = data_test[i + l:i + l + h, k].copy()
                pd[i + l:i + l + h, k] = yhats * (1.0 - _bm) + _gt * _bm
        # save bm and pd to log dir
        np.savez(self._log_dir + "binary_matrix_and_pd", bm=bm, pd=pd)
        predicted_data = scaler.inverse_transform(_pd)
        ground_truth = scaler.inverse_transform(data_test[:_pd.shape[0]])
        np.save(self._log_dir+'pd', predicted_data)
        np.save(self._log_dir+'gt', ground_truth)
        # save metrics to log dir
        error_list = utils.cal_error(ground_truth.flatten(), predicted_data.flatten())
        utils.save_metrics(error_list, self._log_dir, self._alg_name)

    def _predict(self, source):
        states_value = self.encoder_model.predict(source)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self._output_dim))

        # target_seq[0, 0, 0] = source[0, -1, 0]

        yhat = np.zeros(shape=(self._horizon+1, 1),
                        dtype='float32')
        for i in range(self._horizon + 1):
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
            output_tokens = output_tokens[0, -1, 0]
            yhat[i] = output_tokens

            target_seq = np.zeros((1, 1, self._output_dim))
            target_seq[0, 0, 0] = output_tokens

            # Update states
            states_value = [h, c]
        return yhat[-self._horizon:]

    def load(self):
        self.model.load_weights(self._log_dir + 'best_model.hdf5')

    def _save_model_history(self, model_history):
        loss = np.array(model_history.history['loss'])
        val_loss = np.array(model_history.history['val_loss'])
        dump_model_history = pd.DataFrame(index=range(loss.size),
                                          columns=['epoch', 'loss', 'val_loss', 'train_time'])

        dump_model_history['epoch'] = range(loss.size)
        dump_model_history['loss'] = loss
        dump_model_history['val_loss'] = val_loss

        if self._time_callback.times is not None:
            dump_model_history['train_time'] = self._time_callback.times

        dump_model_history.to_csv(self._log_dir + 'training_history.csv', index=False)

    def _plot_training_history(self, model_history):
        import matplotlib.pyplot as plt

        plt.plot(model_history.history['loss'], label='loss')
        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + '[loss]{}.png'.format(self._alg_name))
        plt.legend()
        plt.close()

        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + '[val_loss]{}.png'.format(self._alg_name))
        plt.legend()
        plt.close()

    def plot_models(self):
        plot_model(model=self.model, to_file=self._log_dir + '/model.png', show_shapes=True)


    def plot_series(self):
        from matplotlib import pyplot as plt
        preds = np.load(self._log_dir+'pd.npy')
        gt = np.load(self._log_dir+'gt.npy')

        for i in range(preds.shape[1]):
            plt.plot(preds[:, i], label='preds')
            plt.plot(gt[:, i], label='gt')
            plt.legend()
            plt.savefig(self._log_dir + '[result_predict]series_{}.png'.format(str(i+1)))
            plt.close()