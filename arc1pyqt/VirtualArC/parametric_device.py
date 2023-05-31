import numpy as np
import collections

class ParametricDevice:

    def __init__(self, Ap, An, a0p, a1p, a0n, a1n, tp, tn):
        self.Ap = Ap
        self.An = An
        self.a0p = a0p
        self.a1p = a1p
        self.a0n = a0n
        self.a1n = a1n
        self.tp = tp
        self.tn = tn
        self.Rmem = 0

    def initialise(self, Rinit):
        self.Rmem = Rinit

    def hstep(self, param):
        return 0 if param <= 0 else 1

    def r_V(self, V):
        if V > 0:
            return self.a0p + self.a1p*V
        else:
            return self.a0n + self.a1n*V

    def f_V(self, R, V):
        if V > 0:
            return self.hstep(self.r_V(V) - R)*(self.r_V(V) - R)*(self.r_V(V) - R)
        else:
            return self.hstep(R - self.r_V(V))*(R - self.r_V(V))*(R - self.r_V(V))

    def s_V(self, V):
        if V > 0:
            return self.Ap * (-1 + np.exp(np.abs(V)/self.tp))
        else:
            return self.An * (-1 + np.exp(np.abs(V)/self.tn))

    def step_dt(self, Vm, dt):
        dR = self.s_V(Vm) * self.f_V(self.Rmem, Vm) * dt
        self.Rmem = self.Rmem + dR
