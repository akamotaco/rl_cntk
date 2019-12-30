import cntk as C
import numpy as np
from scipy.stats import multinomial

import cntk_expansion

class MultivariateNormalDiag():
    def __init__(self, loc, scale_diag):
        self.loc = np.array(loc)
        self.scale = np.array(scale_diag) * np.eye(self.loc.shape[0])

        self.loc, self.scale = self.loc.astype(np.float32), self.scale.astype(np.float32)
        self.shape = self.loc.shape
        self.mvn_pdf = C.mvn_pdf(C.constant(self.loc, name='loc'),
                                 C.constant(self.scale, name='scale'))
        self.mvn_log_prob = C.mvn_log_prob(C.constant(self.loc, name='loc'),
                                 C.constant(self.scale, name='scale'))
    def size(self):
        return self.loc.shape
    def sample(self, count):
        return np.random.multivariate_normal(self.loc, self.scale, count)
    def pdf(self, x):
        return self.mvn_pdf(x)
    def log_prob(self, x):
        return self.mvn_log_prob(x)

class Categorical():
    def __init__(self, p, eps=1e-7):
        if isinstance(p, (C.Variable, C.Function)):
            self.p = C.squeeze(p)
        else:
            self.p = C.Constant(np.squeeze(p))

        self.eps = C.Constant(eps, name='eps')
        self.c = self.p.shape[0]

        self.prob = self.p/(self.eps+C.reduce_sum(self.p))
        self.logits = C.log(self.prob)
        self.accum_prob = self.prob@C.Constant((1-np.tri(self.prob.shape[-1],k=-1)))

        p_log_p = self.logits * self.prob
        self._entropy = -C.reduce_sum(p_log_p)

        dist = C.input_variable(1, name='category index')
        # method 1
        self._log_prob = C.log(C.reduce_sum(self.prob * C.one_hot(dist, self.c)))

        # method 2
        # one_hot = C.equal(C.Constant(range(self.c)), dist)
        # self._log_prob = C.log(C.reduce_sum(self.prob * one_hot))

    def sample(self,  n=1):
        samples = C.random.uniform((n,1))
        indcies = C.argmax(C.greater(self.accum_prob-samples,0),axis=1)
        return C.squeeze(indcies)

    def log_prob(self):
        return self._log_prob
    
    def entropy(self):
        return self._entropy
