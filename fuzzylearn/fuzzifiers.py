import math
import numpy as np

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

class BaseFuzzifier:
    def __init__(self, xs=None, mus=None):
        self.xs = xs
        self.mus = mus

    def get_r_to_mu(self, SV_square_distance, sample,\
                    estimated_square_distance_from_center):
        raise NotImplementedError(
        'the base class does not implement get_fuzzified_membership method')

    def get_fuzzified_membership(self, SV_square_distance, sample,\
                 estimated_square_distance_from_center,
                 return_profile=False):
        r_to_mu = self.get_r_to_mu(SV_square_distance, sample,\
                    estimated_square_distance_from_center)
        def estimated_membership(x):
            r = estimated_square_distance_from_center(np.array(x))
            return r_to_mu(r)

        result = [estimated_membership]

        if return_profile:
            rdata = list(map(estimated_square_distance_from_center, self.xs))
            rdata_synth = np.linspace(0, max(rdata)*1.1, 200)
            estimate = list(map(r_to_mu, rdata_synth))
            result.append([rdata, rdata_synth, estimate, SV_square_distance])

        return result

class CrispFuzzifier(BaseFuzzifier):
    def __init__(self, xs=None, mus=None):
        super().__init__(xs, mus)
        
        self.name = 'Crisp'
        self.latex_name = '$\\hat\\mu_{\\text{crisp}}$'

    def get_r_to_mu(self, SV_square_distance, sample,
                 estimated_square_distance_from_center):
        '''Maps distance from center to membership fitting a function
           having the form r -> 1 if r < r_crisp else 0  '''

        def r_to_mu_prototype(r, r_crisp):
            result = np.ones(len(r))
            result[r > r_crisp] = 0
            return result

        rdata = np.fromiter(map(estimated_square_distance_from_center,
                                self.xs),
                            dtype=float)
        popt, _ = curve_fit(r_to_mu_prototype, rdata, self.mus)
                            # bounds=((0,), (np.inf,)))

        if popt[0] < 0:
            raise ValueError('Profile fitting returned a negative parameter')
        return lambda r: r_to_mu_prototype([r], *popt)[0]

    def __repr__(self):
        return 'CrispFuzzifier({}, {})'.format(self.xs, self.mus)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True

class LinearFuzzifier(BaseFuzzifier):
    def __init__(self, xs=None, mus=None, profile='infer'):
        super().__init__(xs, mus)
        self.name = 'Linear'
        self.latex_name = '$\\hat\\mu_{\\text{lin}}$'
        if profile not in ['infer', 'fixed']:
            raise ValueError("'profile' parameter should be either equal to "
                             "'infer' or 'fixed'")
        self.profile = profile

    def get_r_to_mu(self, SV_square_distance, sample,
                    estimated_square_distance_from_center):
        '''Maps distance from center to membership.
           If self.profile='infer' this is done fitting a function
           having the form r -> 1 if r < r_crisp else l(r) where l
           is a linear decreasing function clipped to zero.
           If self.profile='fixed' l is bounded to contain the
           point (SV_square_distance, 0.5).'''


        rdata = np.fromiter(map(estimated_square_distance_from_center,
                                self.xs),
                            dtype=float)

        r_1_guess = np.median([estimated_square_distance_from_center(x)
                               for x, mu in zip(self.xs, self.mus)
                               if mu>=0.99])

        if self.profile == 'fixed':
            def r_to_mu_prototype(r, r_1):
                def r_to_mu_single(rr):
                    r_05 = SV_square_distance
                    res = 1 - 0.5 * (rr - r_1) / (r_05 - r_1)
                    if rr < r_1:
                        return 1
                    elif res < 0:
                        return 0
                    else:
                        return res

                return [r_to_mu_single(rr) for rr in r]

            popt, _ = curve_fit(r_to_mu_prototype,
                                rdata, self.mus,
                                p0=(r_1_guess,),
                                bounds=((0,), (np.inf,)))
        elif self.profile == 'infer':
            def r_to_mu_prototype(r, r_1, r_0):
                def r_to_mu_single(rr):
                    res = 1 - (r_1 - rr) / (r_1 - r_0)
                    if rr < r_1:
                        return 1
                    elif rr > r_0:
                        return 0
                    else:
                        return res
                return [r_to_mu_single(rr) for rr in r]

            popt, _ = curve_fit(r_to_mu_prototype,
                                rdata, self.mus,
                                p0=(r_1_guess, 10*SV_square_distance),
                                bounds=((0, 0), (np.inf, np.inf,)))
        else:
            raise ValueError('This should never happen.'
                             ' Check LinearFuzzifier constructor.')
        if min(popt) < 0:
            raise ValueError('Profile fitting returned a negative parameter')

        return lambda r: r_to_mu_prototype([r], *popt)[0]



    def __repr__(self):
        return "LinearFuzzifier({}, {}, profile='{}')".format(self.xs, self.mus, self.profile)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True


class ExponentialFuzzifier(BaseFuzzifier):
    def __init__(self, xs=None, mus=None, profile='infer', alpha=None):
        super().__init__(xs, mus)
        if profile not in ['infer', 'fixed', 'alpha']:
            raise ValueError("'profile' parameter should be either equal to "
                             "'infer', 'fixed' or 'alpha'")
        self.profile = profile
        if self.profile == 'alpha':
            self.name = 'Exponential({})'.format(alpha)
            self.latex_name = \
                      r'$\hat\mu_{{\text{{exp}},{:.3f}}}$'.format(alpha)
            if alpha is None:
                raise ValueError("'alpha' must be set to a float when"
                                 "'profile' is 'alpha'")
            self.alpha = alpha
        else:
            self.name = 'Exponential'
            self.latex_name = r'$\hat\mu_{\text{nexp}}$'


    def get_r_to_mu(self, SV_square_distance, sample,
                 estimated_square_distance_from_center):

        r_1_guess = np.median([estimated_square_distance_from_center(x)
                               for x, mu in zip(self.xs, self.mus)
                               if mu>=0.99])
        s_guess = (SV_square_distance - r_1_guess) / np.log(2)

        rdata = np.fromiter(map(estimated_square_distance_from_center,
                                self.xs),
                            dtype=float)

        if self.profile == 'fixed':
            def r_to_mu_prototype(r, r_1):
                def r_to_mu_single(rr):
                    s = (SV_square_distance - r_1) / np.log(2)
                    if rr < r_1:
                        return 1
                    else:
                        return np.exp(-(rr - r_1) / s) 
                return [r_to_mu_single(rr) for rr in r]

            popt, _ = curve_fit(r_to_mu_prototype, rdata, self.mus,
                                p0=(r_1_guess,))
                                #bounds=((0,), (np.inf,)))
            return lambda r: r_to_mu_prototype([r], *popt)[0]

        elif self.profile == 'infer':
            def r_to_mu_prototype(r, r_1, s):
                def r_to_mu_single(rr):
                    if rr < r_1:
                        return 1
                    else:
                        return np.exp(-(rr - r_1) / s) 
                return [r_to_mu_single(rr) for rr in r]

            popt, _ = curve_fit(r_to_mu_prototype, rdata, self.mus,
                                #p0=(r_1_guess, s_guess),
                                bounds=((0, 0), (np.inf, np.inf)))

            return lambda r: r_to_mu_prototype([r], *popt)[0]
            #min_radius = np.max([estimated_square_distance_from_center(x)
            #                     for x, mu in zip(self.xs, self.mus) if mu==1])
            #def r_to_mu_prototype(x, s):
            #    result = np.ones(len(x))

            #    indices_smooth = (x > min_radius)
            #    result[indices_smooth] = \
            #                np.exp(-(x[indices_smooth] - min_radius) / s)
            #    return result

        elif self.profile == 'alpha':
            sample = map(estimated_square_distance_from_center, sample)

            q = np.percentile([s-SV_square_distance
                               for s in sample if s > SV_square_distance],
                              100*self.alpha)
            ssd = SV_square_distance

            def r_to_mu(r):
                return 1 if r <= ssd \
                         else np.exp(np.log(self.alpha)/q * (r - ssd))
        
            return r_to_mu
        else:
            raise ValueError('This should not happen. Check the constructor')


    def __repr__(self):
        return "ExponentialFuzzifier({}, {}, profile='{}', alpha={})".format(self.xs, self.mus, self.profile, self.alpha)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other) and self.p == other.p

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True


class QuantileConstantPiecewiseFuzzifier(BaseFuzzifier):
    def __init__(self, xs=None, mus=None):
        self.name = 'QuantileConstPiecewise'
        self.latex_name = '$\\hat\\mu_{\\text{qconst}}$'
        self.xs = xs
        self.mus = mus

    def get_r_to_mu(self, SV_square_distance, sample,
                 estimated_square_distance_from_center):
        sample = map(estimated_square_distance_from_center, sample)
        external_dist = [s-SV_square_distance
                         for s in sample if s > SV_square_distance]
        m = np.median(external_dist)
        q1 = np.percentile(external_dist, 25)
        q3 = np.percentile(external_dist, 75)

        def r_to_mu(r):
            return 1 if r <= SV_square_distance \
                     else 0.75 if r <= SV_square_distance + q1 \
                     else 0.5 if r <= SV_square_distance + m \
                     else 0.25 if r <= SV_square_distance + q3 \
                     else 0

        return r_to_mu

    def __repr__(self):
        return 'QuantileConstantPiecewiseFuzzifier({}, {})'.format(self.xs, self.mus)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True

class QuantileLinearPiecewiseFuzzifier(BaseFuzzifier):
    def __init__(self, xs=None, mus=None):
        self.name = 'QuantileLinPiecewise'
        self.latex_name = '$\\hat\\mu_{\\text{qlin}}$'
        self.xs = xs
        self.mus = mus

    def get_r_to_mu(self, SV_square_distance, sample,
                 estimated_square_distance_from_center):
        sample = np.fromiter(map(estimated_square_distance_from_center,
                                 sample),
                             dtype=float)
        external_dist = [s-SV_square_distance
                         for s in sample if s > SV_square_distance]
        m = np.median(external_dist)
        q1 = np.percentile(external_dist, 25)
        q3 = np.percentile(external_dist, 75)
        mx = np.max(sample) - SV_square_distance

        def r_to_mu(r):
            ssd = SV_square_distance
            return 1 if r <= ssd \
                 else (-r + ssd)/(4*q1) + 1 if r <= ssd + q1 \
                 else (-r + ssd + q1)/(4*(m-q1)) + 3.0/4 if r <= ssd + m \
                 else (-r + ssd + m)/(4*(q3-m)) + 1./2 if r <= ssd + q3 \
                 else (-r + ssd + q3)/(4*(mx-q3)) + 1./4 if r <= ssd+mx \
                 else 0

        return r_to_mu

    def __repr__(self):
        return 'QuantileLinearPiecewiseFuzzifier({}, {})'.format(self.xs, self.mus)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.__repr__())

    def __nonzero__(self):
        return True


