from .tacotron import Tacotron
# models / tacotron.py / Tacotron 클래스

def create_model(name, hparams):
    if name == 'tacotron':
        return Tacotron(hparams)
    else:
        raise Exception('Unknown model' + name) 
