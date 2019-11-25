import argparse
import os
import sys
import yaml
from model.encoder_decoder_supervisor import EncoderDecoder
from lib import utils
import numpy as np

def print_info(mode, config):
    print('----------------------- INFO -----------------------')

    print('|--- MODE:\t{}'.format(mode))
    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- BASE_DIR:\t{}'.format(config['base_dir']))
    print('|--- LOG_LEVEL:\t{}'.format(config['log_level']))
    print('|--- GPU:\t{}'.format(config['gpu']))

    print('----------------------- DATA -----------------------')
    print('|--- BATCH_SIZE:\t{}'.format(config['data']['batch_size']))
    print('|--- TEST_SIZE:\t{}'.format(config['data']['test_size']))

    print('----------------------- MODEL -----------------------')
    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- INPUT_DIM:\t{}'.format(config['model']['input_dim']))
    print('|--- VERIFIED_PERCENTAGE:\t{}'.format(config['model']['verified_percentage']))
    print('|--- L1_DECAY:\t{}'.format(config['model']['l1_decay']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}'.format(config['model']['rnn_units']))
    print('|--- RNN_LAYERS:\t{}'.format(config['model']['rnn_layers']))

    if mode == 'train':
        print('----------------------- TRAIN -----------------------')
        print('|--- DROPOUT:\t{}'.format(config['train']['dropout']))
        print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
        print('|--- OPTIMIZER:\t{}'.format(config['train']['optimizer']))
        print('|--- PATIENCE:\t{}'.format(config['train']['patience']))

    else:
        print('----------------------- TEST -----------------------')
        print('|--- RUN_TIMES:\t{}'.format(config['test']['run_times']))

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes' and infor_correct != 'YES' and infor_correct != 'Y':
        raise RuntimeError('Information is not correct!')


def train(config):
    lstm_ed = EncoderDecoder(is_training=True, **config)
    lstm_ed.plot_models()
    lstm_ed.train()


def test(config):
    lstm_ed = EncoderDecoder(is_training=False, **config)
    lstm_ed.test()
    lstm_ed.plot_series()


def evaluate(config):
    lstm_ed = EncoderDecoder(is_training=False, **config)
    lstm_ed.evaluate()

if __name__ == '__main__':
    np.random.seed(0)
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_file', default='config/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    print_info(args.mode, config)

    if args.mode == 'train':
        train(config)

    elif args.mode == 'evaluate' or args.mode == 'evaluation':
        evaluate(config)

    elif args.mode == "test":
        test(config)
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
