# 텐서플로우에서 멀티쓰레딩 구현하기
# 멀티쓰레딩을 이용하면 여러 개의 쓰레드가 비동기적으로(asynchronous) 병렬연산(parallel)을
# 수행해서 대규모 연산을 훨신 빠른 시간에 수행할 수 있다.

# 큐(queue)를 이용한 멀티쓰레드(Multithread) 연산 수행
# 텐서플로우 에서 그래프를 실행할 때 사용하는 Session 객체는 기본적으로 멀티쓰레드 화 되어 있음
# 따라서 비동기적 연산을 위해 큐를 생성하고 아래 코드와 같이 여러 개의 쓰레드를 생성해서 
# 각각의 쓰레드가 sess.run을 실행하면 간단하게 멀티쓰레드 연산을 수행할 수 있다.

import tensorflow as tf
import threading
import time

# 세션 실행
sess = tf.InteractiveSession()

# 사이즈 100 큐를 생성하고 enqueue 노드를 정의
# 입력동작은 enqueue, 출력 동작은 dequeue

gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
# FIFOQueue -> First in First out, 선입선출 순서로 처리

enqueue = queue.enqueue(gen_random_normal)

# 10개의 임의의 값을 enqueue하는 add 함수를 정의
def add():
    for i in range(10):
        sess.run(enqueue)

# 10개의 쓰레드를 만들고 각각의 쓰레드가 병렬로(parallel) add 함수를 비동기적(asychronous) 실행
threads = [threading.Thread(target=add, args=()) for i in range(10)]
for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.001)
print(sess.run(queue.size()))
time.sleep(0.001)
print(sess.run(queue.size()))