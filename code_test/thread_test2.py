# tf.train.Coordinator를 이용한 쓰레드 컨트롤
# 1에서 간단한 멀티쓰레딩 예제를 보았지만 실제 상황에서는 쓰레드를 정밀하게
# 컨트롤 할 수 있어야 한다. 예를 들어 쓰레드를 잘못 사용할 경우 좀비 쓰레드가 생성될 수 있다.
# 따라서 텐서플로우에는 쓰레드들을 컨트롤 할 수 있는 tf.train.Coordinator API를 제공한다.
# tf.train.Coordinator를 선언하고 Coordinator에 쓰레드들을 join 해주면 
# Coordinator의 should_stop() 함수를 호출해서 쓰레드가 멈춰야 하는 상황인지 체크할 수 있고
# request_stop() 함수를 호출해서 모든 쓰레드를 한번에 정지 시킬 수 있다.

import tensorflow as tf
import threding
import time


coord = tf.train.Coordinator()
