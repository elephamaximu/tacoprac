import argparse
# 파이썬 모듈 --명령행 옵션, 인자와 부속 명령을 위한 파서

import traceback
# 파이썬 모듈 --Print or retrieve a stack traceback

import time
from datetime import datetime
import math
import os
import tensorflow as tf

from datasets.datafeeder import DataFeedeer
# datasets 모듈 / datafeeder.py / DataFeeder 클래스

from hparams import hparams, hparams_debug_string
# hparams.py / hparams 변수 <- tf.contrib.training.HParams 클래스 = HParams 클래스의 객체 , 
# hparams.py / hparams_debug_string 함수

from models import create_model
# models 모듈 / __init__.py / create_model 함수

from text import sequence_to_text
# text 모듈 / __init__.py / sequence_to_text 함수

from util import audio, infolog, plot, ValueWindow
# util 모듈 / audio.py, infolog.py, plot.py, __init__.py / ValueWindow 클래스

log = infolog.log
# util 모듈 / infolog.py / log 함수
# def log(msg):

def add_stats(model):
    with tf.variable_scope('stats') as scope:
        # Class variable_scope
        # Defined in tensorflow/python/ops/variable_scope.py
        # A context manager for defining ops that creates variables(layers)
        tf.summary.histogram('linear_outputs', model.linear_outputs)
        tf.summary.histogram('linear_targets', model.linear_targets)
        tf.summary.histogram('mel_outputs', model.mel_outputs)
        tf.summary.histogram('mel_targets', model.mel_targets)
        tf.summary.scalar('loss_mel', model.mel_loss)
        tf.summary.scalar('loss_linear', model.linear_loss)
        tf.summary.scalar('learning_rate', model.learning_rate)
        tf.summary.scalar('loss', model.loss)
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram('gradient_norm', gradient_norms)
        tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
        return tf.summary.merge_all()

def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def train(log_dir, args):
    # log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name) 아래에 변수 정의
    # args = parser.parse_args() -> argparse 정의 인자들 매개변수 전달
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    # os.path.join :
    # 경로를 병합하여 새 경로 생성 
    # 예) os.path.join('C:\Tmp', 'a', 'b')
    # -> "C:\Tmp\a\b"
    input_path = os.path.join(args.base_dir, args.input)
    # args.base_dir: default=os.path.expanduser('~/dev/tacoprac')
    # args.input: default='training/train.txt'

    log('Checkpoint path: %s' % checkpoint_path)
    log('Loading training data from: %s' % input_path)
    log('Using model: %s' % args.model)
    log(hparams_debug_string())
    # log = infolog.log
    # util / infolog.py / 'msg' 인자 전달 

    # Set up DataFeeder:
    coord = tf.train.Coordinator()
    # tf.train.Coordinator():
    # class Coordinator
    # Defined in tensorflow/python/training/coordinator.py
    # help multiple threads stop together and report exceptions to a program that waits for them to stop
    # 

    with tf._variable_scope('datafeeder') as scope:
        feeder = DataFeedeer(coord, input_path, hparams)
        # DataFeeder --> # datasets 모듈 / datafeeder.py / DataFeeder 클래스
        # 의 객체 feeder 생성
        # class DataFeeder(threading.Thread):
        # --> Lib/threading.py의 Thread 클래스 상속
        # DataFeeder 안에 init 함수와 인자 :
        # def __init__(self, coordinator, metadata_filename, hparams):

    # Set up model:

    # 신경망 모델 구성에 앞서 모델을 저장할 때 쓸 변수 하나 만들어보자.
    # 이 변수는 학습에서 직접 사용하지 않고 학습 횟수를 카운트 하며 추후 데이터 저장시 파일명에 영향을 미치게 된다.
    # 학습에 관여하지 않는 변수를 선언하기 위해 trainable = False라는 옵션을 준다.

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # 이어서 우리가 흔히 알고 있는 여러 계층을 가진 학습 모델을 작성한다.
    # 여러 계층의 모델을 작성하고 마지막 부분에 위에 작성한 global_step 변수를 
    # optimizer.minimize 함수에 매개 변수로 넘겨주자
    # train_op = optimizer.minimize(cost, global_step=global_step)
    # 이와 같이 구성하면 최적화 함수가 학습용 변수들을 최적화 할 때마다 global_step 변수의 값이 1 씩 증가한다.

    with tf.variable_scope('model') as scope:
        model = create_model(args.model, hparams)
        # models 모듈/ __init__.py / create_model 함수
        # def create_model(name, hparams):
        #       return Tacotron(hparams) ---> tocotron.py의 Tacotron 클래스를 반환
        #   if name == 'tacotron': -->  아래 args.model 설정에서 
        # parser.add_argument('--model', default='tacotron') default='tacotron'이므로
        # tacotron이 넘어간다.
        # 따라서 name == tacotron 이므로 tocotron.py의 Tacotron 클래스를 반환하므로
        # 변수 model은 tacotron.py의 Tacotron 클래스를 담고 있다.
       
        model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.linear_targets)
        # models 모듈 / tacotron.py / Tacotron 클래스 / initialize 메소드
        # def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None):
        # 파라미터로 넘어가는 feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.linear_targets :
        # datasets 모듈 / datafeeder.py / DataFeeder(threading.Thread) 클래스
        # 밑에 / _prepare_batch(batch, outputs_per_step) 메서드/ 의 변수들

        # inputs = _prepare_inputs([x[0] for x in batch])
        # input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
        # mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
        # linear_targets = linear_targets = _prepare_targets([x[2] for x in batch], outputs_per_step)

        model.add_loss()
        # models 모듈 / tacotron.py / Tacotron 클래스 / add_loss 메소드
        # def add_loss(self):

        model.add_optimizer(global_step)
        # models 모듈 / tacotron.py / Tacotron 클래스 / add_optimizer 메소드
        # def add_optimizer(self, global_step):
        
        stats = add_stats(model)
        # train.py / def add_stats(model) 함수 --> 텐서보드 그래프

    # Bookkeeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    # util 모듈 / __init__.py / ValueWindow 클래스
    # Class ValueWindow():
    
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
    # Class Saver
    # Defined in tensorflow/python/training/saver.py
    # Saves and restore variables
    # The Saver class adds ops to save and restore variables to and from checkpoints
   
    # Savers can automatically number checkpoint filenames with a provided counter
    # This lets you keep multiple checkpoints at different steps while training a model
    # For example you can number the checkpoint filenames with the training step number
    # To avoid filling up disks, savers manage checkpoint files automatically
    # For example, they can keep only the N most recent files, or one checkpoint for every N hours of training
    # You number checkpoint filenames by passing a value to the optional global_step argument to save()
    '''
    saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
    saver.save(sess, 'my-model', global_step=1000) => filename: 'my-model-1000'
    '''
    # max_to_keep : indicates the maximum number of recent checkpoint files to keep. As new files are created,
    # older files are deleted. if None on 0, no checkpoints are deleted from the filesystem but only the last
    # one is kept in the checkpoint file. Defaults to 5 (that is, the 5 most recent checkpoint files are kept)
    # keep_checkpoint_every_n_hours :  In addition to keeping the most recent max_to_keep checkpoint files,
    # you might want to keep one checkpoint file for every N hours of training. 
    # This can be useful if you want to later analyze how a model progressed during a long training session.
    # keep_checkpoint_every_n_hours=2 : ensures that you keep one checkpoint file for every 2hours of training.
    # The default value of 10,000 hours effectively disables the feature

    # Train!!!
    with tf.Session() as sess:
        try:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
        
            if args.restore_step:
                # Restore from a checkpoint if the user requested it
                restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
                saver.restore(sess, restore_path)
                # restore(sess, save_path)
                log('Resuming from checkpoint: %s' % restore_path)

            else:
                log('Starting new training run')

            feeder.start_in_session(sess)

            while not coord.should_stop():
                # tf.train.Coordinator.should_stop() 함수
                #

                message = 
                log(message)

                if loss > 100 or math.isnan(loss):
                    log()

                if step % args.summary_interval == 0:
                    log()

                if step % args.checkpoint_interval == 0:
                    log()



        except Exeption as e:
            log('Exiting dut to exeption: %s' %e)
            traceback.print_exc()
            coord.request_stop(e)
            # coord = tf.train.Coordinator()
            # Any of the threads can call coord.request_stop() to ask for all the threads to stop
            # To cooperate with the requests, each thread must check for coord.should_stop() on a 
            # regular basis.
            # coord.should_stop() returns True as soon as coord.request.stop() has been called.
  
def main():
  
  # 파서 만들기
  # argparse를 사용하는 첫 번째 단계는 ArgumentParser 객체를 생성하는 것입니다:
  # parser = argparse.ArgumentParser()
  # ArgumentParser 객체는 명령행을 파이썬 데이터형으로 파싱하는데 필요한 모든 정보를 담고 있습니다.
 
  parser = argparse.ArgumentParser()
  
  # 인자 추가하기
  # ArgumentParser에 프로그램 인자에 대한 정보를 채우려면 add_argument() 메서드를 호출하면 됩니다.
  # 일반적으로 이 호출은 ArgumentParser에게 명령행의 문자열을 객체로 변환하는 방법을 알려줍니다.
  # 이 정보는 저장되고, parse_args() 가 호출될 때 사용됩니다. 

  parser.add_argument('--base_dir', default=os.path.expanduser('~/dev/tacoprac'))
  # os.path.expanduser(path) :
  # 입력받은 경로안의 "~를 현재 사용자 디렉토리의 절대경로로 대체합니다.

  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for loggin. Defaults to model name.')
  parser.add_argument('--hparams', default='', 
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')

  # hparams.parse(args.hparams)

  # 인자 파싱하기
  # ArgumentParser는 parse_args() 메서드를 통해 인자를 파싱합니다. 
  args = parser.parse_args()

  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  run_name = args.name or args.model
  os.makedirs(log_dir, exist_ok=True)
  # exist_ok=True  -> mkdir -p와 같은 기능을 한다

  infolog.

  hparams.parse(args.hparams)
  # Override hyperparameters values by parsing the command line
  
  # main() 함수에서 train 함수 호출하여 train 진행
  train(log_dir, args)

  if __name__ =='__main__':
      main()

    # if __name__ == '__main__은 인터프리터에서 직접 실행했을 경우에만
    # if문 내의 코드를 돌리라는 명령