import numpy as np

def epsilon_add(reference:list, computed:list):
    """
     :param reference: list of reference pareto-front values, list of tuples
     :param computed: list of the computed pareto-front values, list of tuples
     :return: additive unary epsilon indicator

     .. math::
        I_{\epsilon}(A,B) = max_{x^2 \in B} min_{x^1 \in A} max_{1 \leq i \leq m} f_i(x^1) - f_i(x^2)

     [1] E. Zitzler, E. Thiele, L. Laummanns, M., Fonseca, C., and Grunert da Fonseca. V (2003):
        Performance Assessment of Multiobjective Optimizers: An Analysis and
        Review. The code is the a Java version of the original metric implementation by Eckart Zitzler.

     [2] Audet, Charles, J. Bigeon, D. Cartier, Sébastien Le Digabel, and Ludovic Salomon.
         "Performance indicators in multiobjective optimization." Optimization Online (2018).
    """


    eps = 0.0
    for ref_val in reference:
        eps_j = np.infty
        for comp_val in computed:
            eps_k = max(np.subtract(comp_val,ref_val))
            eps_j = min(eps_k, eps_j)
        eps = max(eps, eps_j)

    return eps

if __name__ == '__main__':

    # 2d test - euclidean
    ref = [(1., 1.), (2., 0.5), (4., 1. / 4.)]  ### 1/x
    calc = [(1., 1.02), (2., 0.53), (4., 0.26)]

    print(epsilon_add(ref, calc)) # 0.03

