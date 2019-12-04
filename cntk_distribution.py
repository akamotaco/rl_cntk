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
    def __init__(self, p):
        if type(p) is C.Function:
            self.p = C.squeeze(p)
        else:
            self.p = C.Constant(np.squeeze(p))
        self.c = self.p.shape[0]
        self.prob = self.p/C.reduce_sum(self.p)
        self.logits = C.log(self.prob)
        self.accum_prob = self.prob@C.Constant((1-np.tri(self.prob.shape[-1],k=-1)))

    def sample(self,  n=1):
        samples = C.random.uniform((n,1))
        indcies = C.argmax(C.greater(self.accum_prob-samples,0),axis=1)
        return C.squeeze(indcies)

    def log_prob(self, d):
        if type(d) is C.Function:
            from IPython import embed;embed(header='log_prob')
        else:
            # return C.log(C.gather(self.prob, d))
            return C.log(C.reduce_sum(self.prob * C.one_hot(d, self.c)))
    
    def entropy(self):
        p_log_p = self.logits * self.prob
        return -C.reduce_sum(p_log_p)
