import tensorflow as tf

# Hyper Parameter란 신경망 학습을 통해서 튜닝 또는 최적화 해야 하는 주변수가 아니라 학습 진도율이나 일반화 변수처럼,
# 사람들이 선험적인 지식으로 설정하거나 또는 외부 모델의 메커니즘을 통해 자동으로 설정이 되는 변수를 말한다.

# Default hyperparameters:
# Defined in tensorflow/contrib/training/python/training/hparams.py
# Class to hold a set of hyperparameters as name-value pairs.
# A HParams object holds hyperparameters used to build and train a model, such as the number of hidden units
# in a nueral net layer or the learning rate to use when training

# You can override hyperparameter values by calling the parse() method, 
# passing a string of comma separated name=value pairs. 
# This is intended to make it possible to override any hyperparameter values 
# from a single command-line flag to which the user passes 'hyper-param=value' pairs. 
# It avoids having to define one flag for each hyperparameter.

'''
# For example using argparse:
import argparse
parser = argparse.ArgumentParser(description='Train my model.')
parser.add_argument('--hparams', type=str,
                    help='Comma separated list of "name=value" pairs.')
args = parser.parse_args()

def my_program():
  # Create a HParams object specifying the names and values of the
  # model hyperparameters:
  hparams = tf.HParams(learning_rate=0.1, num_hidden_units=100,
                       activations=['relu', 'tanh'])

  # Override hyperparameters values by parsing the command line
  hparams.parse(args.hparams)

  # If the user passed `--hparams=learning_rate=0.3` on the command line
  # then 'hparams' has the following attributes:
  hparams.learning_rate ==> 0.3
  hparams.num_hidden_units ==> 100
  hparams.activations ==> ['relu', 'tanh']
'''
hparams = tf.contrib.training.HParams(
    cleaners='english_cleaners',

    # Audio:
  
    # Model:
  
    # Training:
  
    # Eval:


)

def hparams_debug_string():
    values = hparams.values()
    
    # values() 메서드 :
    # Return the hyperparameter values as a Python dictionary
    
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    retur n 'Hyperparameters:\n' + '\n'.join(hp)