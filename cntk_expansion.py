import autograd.numpy as np
import autograd.numpy.linalg as LA
import cntk as C
from autograd import elementwise_grad, grad, jacobian, holomorphic_grad
from autograd.scipy.stats import multivariate_normal
from cntk import output_variable
from cntk.ops.functions import UserFunction


def __cntk_dot__(x, y):
    return C.reduce_sum(C.element_times(x, y))
C.dot = __cntk_dot__

def __cntk_cov__(m, rowvar: bool = False):
    if len(m.shape) > 2:
        raise ValueError('m has more than 2 dimensions')
    if len(m.shape) < 2:
        m = C.reshape(m, (1, -1))
    if not rowvar and m.shape[0] != 1:
        m = C.transpose(m, [1, 0])

    fact = 1.0 / (m.shape[1] - 1)
    m -= C.reduce_mean(m, axis=1)
    mt = C.transpose(m, [1, 0])
    return fact * C.squeeze(m@mt)
C.cov = __cntk_cov__

def __cntk_cov2__(m):
    m = C.reshape(m, -1)
    m = C.unpack_batch(m)

    m = C.transpose(m, [1, 0])

    count = C.reduce_sum(C.reduce_mean(C.ones_like(m), axis=0))

    fact = 1.0 / (count - 1)
    m -= C.reduce_mean(m, axis=1)
    mt = C.transpose(m,  [1, 0])
    return fact * C.squeeze(m@mt)
C.cov2 = __cntk_cov2__

def __cntk_trace__(m):
    if len(m.shape) != 2:
        raise RuntimeError(f'{m.shape} is not 2 dims')
    if m.shape[0] != m.shape[1]:
        raise RuntimeError(f'{m.shape} is different size')

    _dim = m.shape[0]
    _identity_matrix = C.Constant(np.eye(_dim))
    return C.reduce_sum(m*_identity_matrix)
C.trace = __cntk_trace__

class __cntk_class_det__(UserFunction):
    def __init__(self, arg, name:str='__cntk_class_det__'):
        super(__cntk_class_det__, self).__init__([arg], name=name)

        self.grad = elementwise_grad(LA.det)  # elementwise_grad, jacobian, grad, holomorphic_grad
    
    def forward(self, argument, device=None, output_to_retain=None):
        return argument, LA.det(argument)
    
    def backward(self, state, root_gradients):
        arg = state
        return root_gradients.reshape(root_gradients.shape+(1, 1)) * np.ascontiguousarray(self.grad(arg))

    def infer_outputs(self):
        return [output_variable((), self.inputs[0].dtype, self.inputs[0].dynamic_axes)]
    
    @staticmethod
    def deserialize(inputs, name, state):
        return __cntk_class_det__(inputs[0], name)
def __cntk_det__(m):
    return C.user_function(__cntk_class_det__(m))
C.det = __cntk_det__

class __cntk_class_slogdet__(UserFunction):
    def __init__(self, arg, name:str='__cntk_class_slogdet__'):
        super(__cntk_class_slogdet__, self).__init__([arg], name=name)

        self.grad = elementwise_grad(LA.slogdet)  # elementwise_grad, jacobian, grad, holomorphic_grad
    
    def forward(self, argument, device=None, output_to_retain=None):
        return argument, np.ascontiguousarray(np.stack(LA.slogdet(argument)).T)

    def backward(self, state, root_gradients): # 수정 필요
        arg = state
        return root_gradients.reshape(root_gradients.shape+(1,)) * np.ascontiguousarray(self.grad(arg))

    def infer_outputs(self): # 수정 필요
        return [output_variable((2), self.inputs[0].dtype, self.inputs[0].dynamic_axes)]
    
    @staticmethod
    def deserialize(inputs, name, state):
        return __cntk_class_slogdet__(inputs[0], name)
def __cntk_slogdet__(m):
    return C.user_function(__cntk_class_slogdet__(m))
C.slogdet = __cntk_slogdet__

# class MySigmoid(UserFunction):
#     def __init__(self, arg, name='MySigmoid'):
#         super(MySigmoid, self).__init__([arg], name=name)
#         self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
#         self.grad = grad(self.sigmoid)

#     def forward(self, argument, device=None, outputs_to_retain=None):
#         return argument, self.sigmoid(argument)

#     def backward(self, state, root_gradients):
#         argument = state
#         return root_gradients * self.grad(argument)

#     def infer_outputs(self):
#         return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

#     @staticmethod
#     def deserialize(inputs, name, state):
#         return MySigmoid(inputs[0], name)

class __cntk_class_mvn_pdf__(UserFunction):
    def __init__(self, X, loc, scale, name: str = '__cntk_class_mvn_pdf__'):
        super(__cntk_class_mvn_pdf__, self).__init__([X, loc, scale], name=name)
        self.mvn_pdf = multivariate_normal.pdf

        self.grad = elementwise_grad(self.mvn_pdf)  # elementwise_grad, jacobian, grad, holomorphic_grad

    def forward(self, arguments, device=None, outputs_to_retain=None):
        x, loc, scale = arguments
        return arguments, self.mvn_pdf(x, loc, scale).astype(np.float32).reshape(-1, 1)

    def backward(self, state, root_gradients, variables):
        x, loc, scale = state
        _grad = root_gradients * np.ascontiguousarray(self.grad(x, loc, scale).astype(np.float32))

        for k in variables:
            variables[k] = _grad

    def infer_outputs(self):
        return [output_variable((1), self.inputs[0].dtype, self.inputs[0].dynamic_axes)]
        # return [output_variable((), self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    @staticmethod
    def deserialize(inputs, name, state):
        return __cntk_class_mvn_pdf__(inputs[0], inputs[1], inputs[2], name)

def __cntk_mvn_pdf__(mu, sig):
    @C.Function
    def _(x): return C.user_function(__cntk_class_mvn_pdf__(x, mu, sig))
    return _
C.mvn_pdf = __cntk_mvn_pdf__

class __cntk_class_mvn_log_prob__(UserFunction):
    def __init__(self, X, loc, scale, name: str = '__cntk_class_mvn_log_prob__'):
        super(__cntk_class_mvn_log_prob__, self).__init__([X, loc, scale], name=name)
        self.log_prob = multivariate_normal.logpdf

        self.grad = elementwise_grad(self.log_prob)  # elementwise_grad, jacobian, grad, holomorphic_grad

    def forward(self, arguments, device=None, outputs_to_retain=None):
        x, loc, scale = arguments
        return arguments, self.log_prob(x, loc, scale).astype(np.float32).reshape(-1, 1)

    def backward(self, state, root_gradients, variables):
        x, loc, scale = state
        _grad = root_gradients * np.ascontiguousarray(self.grad(x, loc, scale).astype(np.float32))

        for k in variables:
            variables[k] = _grad

    def infer_outputs(self):
        return [output_variable((1), self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    @staticmethod
    def deserialize(inputs, name, state):
        return __cntk_class_mvn_log_prob__(inputs[0], inputs[1], inputs[2], name)

def __cntk_mvn_log_prob__(mu, sig):
    @C.Function
    def _(x): return C.user_function(__cntk_class_mvn_log_prob__(x, mu, sig))
    return _
C.mvn_log_prob = __cntk_mvn_log_prob__

if __name__ == '__main__':
    q = C.mvn_pdf(C.constant([0, 0]), C.constant([[1, 0], [0, 1]]))(C.input_variable(2, needs_gradient=True))
    q.eval({q.arguments[0]:np.random.normal(size=(100, 2))})
    q.grad({q.arguments[0]:np.random.normal(size=(100, 2))})

    q = C.slogdet(C.input_variable((2,2),needs_gradient=True))
    q.eval({q.arguments[0]:np.array([[[1,2],[3,4]]]*3,np.float32).reshape(3,2,2)})
    q.grad({q.arguments[0]:np.array([[[1,2],[3,4]]]*3,np.float32).reshape(3,2,2)})
