import atexit
# atexit 모듈은 정리 함수를 등록하고 해제하는 함수를 정의합니다.
# 이렇게 등록된 함수는 정상적인 인터프리터 종료 시 자동으로 실행됩니다.
# 

from datetime import datetime
import json
from threading import Thread

def log(msg):
    print(msg)

def _close_logfile():
    global _file

atexit.register(_close_logfile)
# func를 종료 시에 실행 할 함수로 등록합니다.