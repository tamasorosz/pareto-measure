from scipy import spatial
import numpy as np


def gd(reference: list, computed: list, norm='euclidean'):
    """
    :param reference: list of the reference points [(coord_1, coord_2, coord_3,..., coord_n)]
    :return: the sum of the euclidean disctance from the given reference points.

    Possible norms:  ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
                     ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’,
                     ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
                     ‘sqeuclidean’, ‘wminkowski’, ‘yule’.

    Type: Convergence Indicator

    The generational distance performence indicator [1,2,3] measueres the distancee from the
    given reference list. T

    .. math::

        GD(S,P) = \frac{1}{n}(\Sigma_{s \in S} min ||F(s) - F(r)||^2)^{\frac{1}{p}}

    where |S| is the number of the points in a Pareto-set approximation and P a discrete representation
    of the Pareto-front. Generally, p=2. In this case it is equivalent with M1 measure [4].
    When P = 1 it is equivalent with the Gamma-Metric.

    References
    ----------

    .: [1] David A. Van Veldhuizen and David A. Van Veldhuizen.
           Multi-objective evolutionary algorithms: classifications, analyses, and new innovations.
           Technical Report, Evolutionary Computation, 1999.
    .: [2] Audet, Charles, J. Bigeon, D. Cartier, Sébastien Le Digabel, and Ludovic Salomon.
           "Performance indicators in multiobjective optimization." Optimization Online (2018).
    .: [3] https://pymoo.org/misc/performance_indicator.html
    .: [4] E. Zitzler, K. Deb, L. Thiele, Comparison of multiobjective evolutionary algorithms: Empirical
           results, Evolutionary computation 8 (2) (2000) 173–195.

    """

    distances = spatial.distance.cdist(reference, computed, metric=norm)
    minimums = np.nanmin(distances, axis=0)

    return np.sum(minimums) / len(computed)


if __name__ == '__main__':
    # 2d test - euclidean
    ref = [(1., 1.), (2., 1. / 2.), (3., 1. / 3.), (4., 1. / 4.), (5., 1. / 5)]  ### 1/x
    calc = [(1., 1.01), (2., 0.51), (4., 0.26)]

    print(gd(ref, calc))  # result is 0.01

    # 3d test - chebyshev
    ref = [(1., 1., 1.), (2., 1. / 2., 1. / 2.), (3., 1. / 3., 1. / 3.), (4., 1. / 4., 1. / 4.),
           (5., 1. / 5, 1. / 5,)]  ### 1/x
    calc = [(1., 1.01, 1.01), (2., 0.51, 0.51), (4., 0.26, 0.26)]

    print(gd(ref, calc, norm='chebyshev'))  # 0.01

