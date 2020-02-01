import numpy as np
import tqdm
import itertools as it

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state

from fuzzylearn.kernel import GaussianKernel, PrecomputedKernel
from fuzzylearn.fuzzifiers import ExponentialFuzzifier

from warnings import warn

try:
    import gurobipy as gpy
    gurobi_ok = True
except ModuleNotFoundError:
    print('gurobi not available')
    gurobi_ok = False

try:
    import tensorflow as tf
    tensorflow_ok = True
except ModuleNotFoundError:
    print('tensorflow not available')
    tensorflow_ok = False

def chop(x, minimum, maximum, tolerance=1e-4):
    '''Chops a number when it is sufficiently close to the extreme of
an enclosing interval.

Arguments:

- x: number to be possibily chopped
- minimum: left extreme of the interval containing x
- maximum: right extreme of the interval containing x
- tolerance: maximum distance in order to chop x

Returns: x if it is farther than tolerance by both minimum and maximum;
         minimum if x is closer than tolerance to minimum
         maximum if x is closer than tolerance to maximum

Throws:

- ValueError if minimum > maximum or if x does not belong to [minimum, maximum]

'''
    if minimum > maximum:
        raise ValueError('Chop: interval extremes not sorted')
    #if  x < minimum or x > maximum:
    #    raise ValueError('Chop: value not belonging to interval')

    if x - minimum < tolerance:
        x = minimum
    if maximum - x < tolerance:
        x = maximum
    return x

def get_argument_value(key, opt_args, default_args):
    return opt_args[key] if key in opt_args else default_args[key]

def solve_optimization_tensorflow(x, mu, c=1.0, k=GaussianKernel(),
                                  opt_args={}):
    '''Builds and solves the constrained optimization problem on the basis
of the fuzzy learning procedure using the TensorFlow API.

Arguments:

- x: iterable of objects
- mu: iterable of membership values for the objects in x
- c: constant managing the trade-off in joint radius/error optimization
- k: kernel function to be used
- adjustment: diagonal adjustment in order to deal with non PSD matrices
- opt_args: arguments for TensorFlow (currently nothing)

Returns: a lists containing the optimal values for the independent
         variables chis of the problem

Throws:

- ValueError if optimization fails or if tensorflow is not installed

'''

    if not tensorflow_ok:
        raise ValueError('tensorflow not available')

    default_args = {'init': 'fixed',
                    'init_bound': 0.1,
                    'init_val': 0.01,
                    'n_iter': 100,
                    'optimizer': tf.optimizers.Adam(learning_rate=1e-4),
                    'tracker': tqdm.trange}

    init = get_argument_value('init', opt_args, default_args)

    m = len(x)

    if type(init) == str and init == 'fixed':
        init_val = get_argument_value('init_val', opt_args, default_args)
        chis = [tf.Variable(init_val, name='chi_{}'.format(i),
                            trainable=True, dtype=tf.float32)
                for i in range(m)]
    elif type(init) == str and init == 'random':
        l = get_argument_value('init_bound', opt_args, default_args)
        chis = [tf.Variable(ch, name='chi_{}'.format(i),
                            trainable=True, dtype=tf.float32)
                for i, ch in  enumerate(np.random.uniform(-0.1, 0.1, m))]

    elif type(init) == list or type(init) == np.ndarray:
        chis = [tf.Variable(ch, name='chi_{}'.format(i),
                            trainable=True, dtype=tf.float32)
                for i, ch in  enumerate(init)]
    else:
        raise ValueError("init should either be set to 'fixed', "
                         "'random', or to a list of initial values.")

    if type(k) is PrecomputedKernel:
      gram = k.kernel_computations
    else:
      gram = np.array([[k.compute(x1, x2) for x1 in x] for x2 in x])

    def obj():
        penal = 10
        kernels = tf.constant(gram, dtype='float32')

        v = tf.tensordot(tf.linalg.matvec(kernels, chis), chis, axes=1)
        v -= tf.tensordot(chis, [k.compute(x_i, x_i) for x_i in x], axes=1)

        #if adjustment:
        #    v += adjustment * tf.tensordot(chis, chis, axes=1)

        v += penal * tf.math.maximum(0, 1 - sum(chis))
        v += penal * tf.math.maximum(0, sum(chis) - 1)

        if c < np.inf:
            for ch, m in zip(chis, mu):
                v += penal * tf.math.maximum(0, ch - c*m)
                v += penal * tf.math.maximum(0, c*(1-m) - ch)

        return v

    opt = get_argument_value('optimizer', opt_args, default_args)

    n_iter = get_argument_value('n_iter', opt_args, default_args)

    tracker = get_argument_value('tracker', opt_args, default_args)
    if tracker is None:
        tracker = range


    for i in tracker(n_iter):
        #old_chis = np.array([ch.numpy() for ch in chis])
        opt.minimize(obj, var_list=chis)
        #new_chis = np.array([ch.numpy() for ch in chis])

    return [ch.numpy() for ch in chis]


def solve_optimization_gurobi(x,
                              mu,
                              c=1.0,
                              k=GaussianKernel(),
                              opt_args={}):
    '''Builds and solves the constrained optimization problem on the basis
of the fuzzy learning procedure using the gurobi API.

Arguments:

- x: iterable of objects
- mu: iterable of membership values for the objects in x
- c: constant managing the trade-off in joint radius/error optimization
- k: kernel function to be used
- adjustment: diagonal adjustment in order to deal with non PSD matrices
- opt_args: arguments for gurobi ('time_limit' is the time in seconds before
            stopping the optimization process)

Returns: a lists containing the optimal values for the independent
         variables chis of the problem

Throws:

- ValueError if optimization fails or if gurobi is not installed

'''

    if not gurobi_ok:
        raise ValueError('gurobi not available')

    default_args = {'time_limit': 10*60, 'adjustment': 0}
    m = len(x)

    model = gpy.Model('possibility-learn')
    model.setParam('OutputFlag', 0)
    time_limit = get_argument_value('time_limit', opt_args, default_args)
    adjustment = get_argument_value('adjustment', opt_args, default_args)
    model.setParam('TimeLimit', time_limit)

    for i in range(m):
        if c < np.inf:
            model.addVar(name='chi_%d' % i, lb=-c*(1-mu[i]), ub=c*mu[i],
                         vtype=gpy.GRB.CONTINUOUS)

        else:
            model.addVar(name='chi_%d' % i, vtype=gpy.GRB.CONTINUOUS)

    model.update()

    chis = model.getVars()

    obj = gpy.QuadExpr()

    for i, j in it.product(range(m), range(m)):
        obj.add(chis[i] * chis[j], k.compute(x[i], x[j]))

    for i in range(m):
        obj.add(-1 * chis[i] * k.compute(x[i], x[i]))

    if adjustment:
        for i in range(m):
            obj.add(adjustment * chis[i] * chis[i])

    model.setObjective(obj, gpy.GRB.MINIMIZE)

    constEqual = gpy.LinExpr()
    constEqual.add(sum(chis), 1.0)

    model.addConstr(constEqual, gpy.GRB.EQUAL, 1)

    model.optimize()


    if model.Status != gpy.GRB.OPTIMAL:
        raise ValueError('optimal solution not found!')

    return [ch.x for ch in chis]

def solve_optimization(x, mu, c=1.0, k=GaussianKernel(),
                       solve_strategy=solve_optimization_tensorflow,
                       solve_strategy_args={},
                       tolerance=1e-4,
                       adjustment=0):
    '''Builds and solves the constrained optimization problem on the basis
of the fuzzy learning procedure.

Arguments:

- x: iterable of objects
- mu: iterable of membership values for the objects in x
- c: constant managing the trade-off in joint radius/error optimization
- k: kernel function to be used
- tolerance: tolerance to be used in order to clamp the problem solution to
             interval extremes
- adjustment: diagonal adjustment in order to deal with non PSD matrices
- solve_strategy: algorithm to be used in order to numerically solve the
                  optimization problem
- solve_strategy_args: optional parameters for the optimization algorithm

Returns: a lists containing the optimal values for the independent
         variables chis of the problem

Throws:

- ValueError if c is non-positive or if x and mu have different lengths

'''
    if c <= 0:
        raise ValueError('c should be positive')


    mu = np.array(mu)


    chis = solve_strategy(x, mu, c, k, solve_strategy_args)

    chis_opt = [chop(ch, l, u, tolerance)
                for ch, l, u in zip(chis, -c*(1-np.array(mu)), c*np.array(mu))]

    return chis_opt

class FuzzyInductor(BaseEstimator, RegressorMixin):

    def __init__(self, c=1, k=GaussianKernel(),
                 sample_generator=None, fuzzifier=ExponentialFuzzifier,
                 solve_strategy=(solve_optimization_tensorflow, {}),
                 random_state=None,
                 return_vars=False, return_profile=False):
        self.c = c
        self.k = k
        self.sample_generator = sample_generator
        self.fuzzifier = fuzzifier
        self.solve_strategy = solve_strategy
        self.random_state = random_state
        self.return_vars = return_vars
        self.return_profile = return_profile

    def fit(self, X, y, **kwargs):

        check_X_y(X, y)
        self.random_state_ = check_random_state(self.random_state)

        if 'warm_start' in kwargs and kwargs['warm_start']:
            check_is_fitted(self, ['chis_', 'estimated_membership_'])
            self.solve_strategy[1]['init'] = self.chis_

        self.chis_ = solve_optimization(X, y,
                                        self.c, self.k,
                                        self.solve_strategy[0],
                                        self.solve_strategy[1])

        if type(self.k) is PrecomputedKernel:
            self.gram_ = self.k.kernel_computations
        else:
            self.gram_ = np.array([[self.k.compute(x1, x2) for x1 in X]
                                    for x2 in X])
        self.fixed_term_ = np.array(self.chis_).dot(self.gram_.dot(self.chis_))

        def estimated_square_distance_from_center(x_new):
            ret = self.k.compute(x_new, x_new) \
                  - 2 * np.array([self.k.compute(x_i, x_new)
                                  for x_i in X]).dot(self.chis_) \
                  + self.fixed_term_
            return ret
        self.estimated_square_distance_from_center_ = \
                estimated_square_distance_from_center

        self.chi_SV_index_ = [i for i, (chi, mu) in enumerate(zip(self.chis_,
                                                                  y))
                              if -self.c * (1-mu) < chi < self.c * mu]

        #self.chi_SV_index_ = [i for i in range(len(self.chis)_) \
        #        if -self.c*(1-self.mu[i]) < self.chis_[i] < self.c*self.mu[i]]

        chi_SV_square_distance = map(estimated_square_distance_from_center,
                                     X[self.chi_SV_index_])
        chi_SV_square_distance = list(chi_SV_square_distance)
        #chi_SV_square_distance = [estimated_square_distance_from_center(x[i])
        #                          for i in chi_SV_index]

        if len(chi_SV_square_distance) == 0:
            self.estimated_membership_ = None
            self.train_error_ = np.inf
            self.chis_ = None
            self.profile = None
            warn('No support vectors found')
            return self
            #raise ValueError('No support vectors found')

        self.SV_square_distance_ = np.mean(chi_SV_square_distance)
        num_samples = 500

        if self.sample_generator is None:
            self.sample_generator = lambda x: x

        sample = map(self.sample_generator,
                     self.random_state_.random_sample(num_samples))


        fuzzifier = self.fuzzifier(X, y)
        result = fuzzifier.get_fuzzified_membership(
                self.SV_square_distance_,
                sample,
                self.estimated_square_distance_from_center_,
                return_profile=self.return_profile)


        if self.return_profile:
            self.estimated_membership_, self.profile_ = result
        else:
            self.estimated_membership_ = result[0]

        self.train_error_ = np.mean([(self.estimated_membership_(x) - mu)**2
                                    for x, mu in zip(X, y)])

        return self


    def predict(self, X):

        check_is_fitted(self, ['chis_', 'estimated_membership_'])
        X = check_array(X)
        return np.array([self.estimated_membership_(x) for x in X])

    def score(self, X, y):
        if self.estimated_membership_ :
            return -np.mean([(self.estimated_membership_(x) - mu)**2 for x, mu in zip(X, y)])
        else:
            return -np.inf
