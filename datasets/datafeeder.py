

class DataFeedeer(threading.Thread):
# class threading.Thread(group=None, target=None, name=None, args=(), kwargs={}, *,daemon=None)
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
        super(DataFeedeer, self).__init__()