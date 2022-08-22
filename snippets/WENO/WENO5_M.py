"""WENO-M

Ref: Henrick, Andrew K., Tariq D. Aslam, and Joseph M. Powers. 
"Mapped weighted essentially non-oscillatory schemes: achieving 
optimal order near critical points." Journal of Computational Physics 207.2 (2005): 542-567.

"""


def WENO5_M_L(vmm, vm, vo, vp, vpp):
    """WENO5_M_L Reconstruction of the :math:`u_{i-1/2}^+` by WENO5-M
    ::
        
                |___________S0__________|
                |                       |
                |       |___________S1__________|
                |       |                       |
                |       |       |___________S2__________|
              ..|---o---|---o---|---o---|---o---|---o---|...
                | I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|
                                |+
                                i-1/2
        

    Args:
        vmm : :math:`u_{i-2}`
        vm : :math:`u_{i-1}` 
        vo : :math:`u_i`
        vp : :math:`u_{i+1}`
        vpp : :math:`u_{i+2}`
    Returns:
        :math:`u_{i-1/2}^+`
    """
    # Smooth Indicators (Beta factors)
    B0 = 13 / 12 * (vmm - 2 * vm + vo) ** 2 + 1 / 4 * (vmm - 4 * vm + 3 * vo) ** 2
    B1 = 13 / 12 * (vm - 2 * vo + vp) ** 2 + 1 / 4 * (vm - vp) ** 2
    B2 = 13 / 12 * (vo - 2 * vp + vpp) ** 2 + 1 / 4 * (3 * vo - 4 * vp + vpp) ** 2

    # Constants
    d0p = 3 / 10
    d1p = 6 / 10
    d2p = 1 / 10
    epsilon = 1e-40

    # Alpha weights
    alpha0p = d0p / (epsilon + B0) ** 2
    alpha1p = d1p / (epsilon + B1) ** 2
    alpha2p = d2p / (epsilon + B2) ** 2
    alphasump = alpha0p + alpha1p + alpha2p

    # ENO stencils weigths
    w0p = alpha0p / alphasump
    w1p = alpha1p / alphasump
    w2p = alpha2p / alphasump

    g0p = (
        w0p
        * (d0p + d0p * d0p - 3 * d0p * w0p + w0p ^ 2)
        / (d0p * d0p + w0p * (1 - 2 * d0p))
    )
    g1p = (
        w1p
        * (d1p + d1p * d1p - 3 * d1p * w1p + w1p ^ 2)
        / (d1p * d1p + w1p * (1 - 2 * d1p))
    )
    g2p = (
        w2p
        * (d2p + d2p * d2p - 3 * d2p * w2p + w2p ^ 2)
        / (d2p * d2p + w2p * (1 - 2 * d2p))
    )
    gsump = g0p + g1p + g2p

    # Modified weigths
    w0p = g0p / gsump
    w1p = g1p / gsump
    w2p = g2p / gsump

    # Numerical Flux at cell boundary, $u_{i+1/2}^{+}$;
    fluxL = (
        w0p * (-vmm + 5 * vm + 2 * vo) / 6
        + w1p * (2 * vm + 5 * vo - vp) / 6
        + w2p * (11 * vo - 7 * vp + 2 * vpp) / 6
    )

    return fluxL


def WENO5_M_R(vmm, vm, vo, vp, vpp):
    """WENO5_M_L Reconstruction of the :math:`u_{i+1/2}^-` by WENO5-M
    ::
        
                |___________S0__________|
                |                       |
                |       |___________S1__________|
                |       |                       |
                |       |       |___________S2__________|
              ..|---o---|---o---|---o---|---o---|---o---|...
                | I{i-2}| I{i-1}|  I{i} | I{i+1}| I{i+2}|
                                       -|
                                      i+1/2
        

    Args:
        vmm : :math:`u_{i-2}`
        vm : :math:`u_{i-1}` 
        vo : :math:`u_i`
        vp : :math:`u_{i+1}`
        vpp : :math:`u_{i+2}`
    Returns:
        :math:`u_{i+1/2}^-`
    """

    # Smooth Indicators (Beta factors)
    B0 = 13 / 12 * (vmm - 2 * vm + vo) ** 2 + 1 / 4 * (vmm - 4 * vm + 3 * vo) ** 2
    B1 = 13 / 12 * (vm - 2 * vo + vp) ** 2 + 1 / 4 * (vm - vp) ** 2
    B2 = 13 / 12 * (vo - 2 * vp + vpp) ** 2 + 1 / 4 * (3 * vo - 4 * vp + vpp) ** 2

    d0n = 1 / 10
    d1n = 6 / 10
    d2n = 3 / 10
    epsilon = 1e-40

    # Alpha weights
    alpha0n = d0n / (epsilon + B0) ** 2
    alpha1n = d1n / (epsilon + B1) ** 2
    alpha2n = d2n / (epsilon + B2) ** 2
    alphasumn = alpha0n + alpha1n + alpha2n

    # ENO stencils weigths
    w0n = alpha0n / alphasumn
    w1n = alpha1n / alphasumn
    w2n = alpha2n / alphasumn

    g0n = (
        w0n
        * (d0n + d0n * d0n - 3 * d0n * w0n + w0n ^ 2)
        / (d0n * d0n + w0n * (1 - 2 * d0n))
    )
    g1n = (
        w1n
        * (d1n + d1n * d1n - 3 * d1n * w1n + w1n ^ 2)
        / (d1n * d1n + w1n * (1 - 2 * d1n))
    )
    g2n = (
        w2n
        * (d2n + d2n * d2n - 3 * d2n * w2n + w2n ^ 2)
        / (d2n * d2n + w2n * (1 - 2 * d2n))
    )
    gsumn = g0n + g1n + g2n

    # Modified weigths
    w0n = g0n / gsumn
    w1n = g1n / gsumn
    w2n = g2n / gsumn

    fluxR = (
        w0n * (2 * vmm - 7 * vm + 11 * vo) / 6
        + w1n * (-vm + 5 * vo + 2 * vp) / 6
        + w2n * (2 * vo + 5 * vp - vpp) / 6
    )

    return fluxR
