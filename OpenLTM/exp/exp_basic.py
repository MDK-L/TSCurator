from models import timer, timer_xl, timer_ours, moirai, patchtst, linear, alinear, mhlinear


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'linear': linear,
            'alinear': alinear,
            'mhlinear': mhlinear,
            'timer_ours': timer_ours,
            'timer': timer,
            'timer_xl': timer_xl,
            'moirai': moirai,
            'patchtst': patchtst
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
