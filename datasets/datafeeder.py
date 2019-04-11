import numpy as np
import os
import tensorflow as tf
import time
import traceback
from util.infolog import log
import threading

class DataFeedeer(threading.Thread):
# class threading.Thread()
# This constructor should always be called with keyword argument. Arguments are:
# group should be None; reserved for future extension when a ThreadGroup class is implemented
# target is the callable object to be invoked by the run() method. Defaults to None, meaning nothing is called
# name is the thread name. By default, a unique name is constructed of the form "Thread-N" where
# N is a small decimal number
# args is the argument tuple for the target invocation. Default to ()
# kwargs is a dictionary of keyword arguments for the target invocation. Default to {}
# if not None, daemon explicitly sets whether the thread is daemonic. If None(default), the daemonic
# property is inherited from the current thread

# if the subclass overrides the constructor, it must make sure to invoke the base class constructor
# (Thread.__init__()) before doing anything else to the thread

# start()
# run()
# join(timeout=None)

# Thread Objects
# The Thread class represents an activity that is run in a separate thread of control.
# There are two ways to specify the activity:
# by passing a callable object to the constructor, or by overriding the run() method in a subclass.
# No other methods (except for the constructor) should be overridden in a subclass.
# In other words, only override the __init__() and run() methods of this class

    def __init__(self, coordinator, metadata_filename, hparams):
        # train.py 에서 feeder = DataFeedeer(coord, input_path, hparams)로 객체 생성
        super(DataFeedeer, self).__init__()
        self.__coord = coordinator
        # train.py 에서 coord = tf.train.Coordinator()
        self.__hparams = hparams

        self._offset = 0

        # Load metadata:
        self._datadir = os.path.dirname(metadata_filename)
        # os.path.dirname(경로) : 경로 중에서 디렉토리 명만 얻기
        # ex) os.path.dirname("C:/Python35/Scripts/pip.exe")
        # ==> "C:/Python35/Scripts"

        with open(metadata_filename, encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]
            hours = 
            log('')


        
        # Creating queue for buffering data:
        queue = tf.FIFOQueue() 

        # tf.queue.FIFOQueue
        # FIFOQueue 클래스 Inherited From QueueBase
        # Defined in tensorflow/python/ops/data_flow_ops.py.

    def start_in_session(self, session):
        self._session = session
        self.start()

    def run(self):
        try:
            while not self.__coord.should_stop():
                self.__enqueue_next_group()
        except Exception as e:
            traceback.print_exc()
            self.__coord.request_stop(e)