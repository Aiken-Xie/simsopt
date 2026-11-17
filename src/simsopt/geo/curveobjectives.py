from deprecated import deprecated

import numpy as np
from jax import grad
import jax.numpy as jnp

from .jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
from ..objectives.utilities import forward_backward
import simsoptpp as sopp

__all__ = ['CurveLength', 'LpCurveCurvature', 'LpCurveTorsion',
           'CurveCurveDistance', 'CurveSurfaceDistance', 'ArclengthVariation',
           'MeanSquaredCurvature', 'LinkingNumber',
           ]


def _dense_curve_proxy(curve, oversample):
    """A curve-shaped object sharing ``curve``'s own Fourier dofs, sampled
    at ``oversample`` times as many quadrature points.

    Curves are only ever sampled at their own fixed ``numquadpoints`` (a
    curve constructed at, say, 128 points has no way to reveal a sharp local
    feature -- a close approach to another curve, or a curvature spike --
    that happens to fall between those 128 samples). This does not change
    ``curve`` itself, only what a penalty checks it against; returned proxy
    must have its dofs re-synced (``proxy.x = curve.x``, or via
    :func:`_resync_dense_curve_proxy` for a possibly ``RotatedCurve``-wrapped
    one) before every use, since ``curve``'s dofs change during optimization.

    Handles a plain :class:`CurveXYZFourier` or a :class:`RotatedCurve`
    wrapping one (the latter is what coils built via ``coils_via_symmetries``
    actually are); returns ``(outer_proxy, inner_base_proxy)`` where
    ``outer_proxy`` is what to call ``gamma()``/``kappa()``/etc. on and
    ``inner_base_proxy`` is what to resync dofs onto.
    """
    if hasattr(curve, "curve") and hasattr(curve, "rotmat"):
        from .curve import RotatedCurve
        inner_outer, inner_base_proxy = _dense_curve_proxy(curve.curve, oversample)
        flip = bool(np.linalg.det(curve.rotmat) < 0)
        outer_proxy = RotatedCurve(inner_outer, curve._phi, flip)
        return outer_proxy, inner_base_proxy
    from .curvexyzfourier import CurveXYZFourier
    dense = CurveXYZFourier(oversample * len(curve.quadpoints), curve.order)
    dense.x = curve.x
    return dense, dense


def _resync_dense_curve_proxy(curve, inner_base_proxy):
    """Re-sync a proxy built by :func:`_dense_curve_proxy` to ``curve``'s
    current dofs (cheap: just copies the Fourier coefficients)."""
    base = curve
    while hasattr(base, "curve"):
        base = base.curve
    inner_base_proxy.x = base.x


def _rekey_derivative(deriv, from_obj, to_obj):
    """Re-key a :class:`Derivative` from ``from_obj`` to ``to_obj``.

    A :class:`Derivative` returned by e.g. ``curve.dgamma_by_dcoeff_vjp(...)``
    is internally keyed to whichever Optimizable produced it -- normally
    that is exactly the curve whose dofs the caller cares about, so this is
    invisible. But a dense proxy curve built by :func:`_dense_curve_proxy`
    is deliberately *not* part of the real dependency graph (it shares dofs
    with the original curve rather than being an independent parameter), so
    a ``Derivative`` keyed to the proxy (or, for a ``RotatedCurve`` proxy,
    to the proxy's own dense inner base) would silently resolve to zero when
    later evaluated against the real curve's lineage. Re-key it here so it
    resolves correctly.
    """
    if from_obj is to_obj:
        return deriv
    full = deriv(from_obj, as_derivative=True).data[from_obj]
    return Derivative({to_obj: full})


@jit
def curve_length_pure(l):
    """
    Compute the mean of the incremental arclengths along a curve (the curve length).

    Args:
        l (array-like): Array of incremental arclengths along the curve.

    Returns:
        float: The mean arclength (i.e., the curve length).
    """
    return jnp.mean(l)


class CurveLength(Optimizable):
    r"""
    CurveLength is a class that computes the length of a curve, i.e.

    .. math::
        J = \int_{\text{curve}}~dl.

    """

    def __init__(self, curve):
        self.curve = curve
        self.dJ_dl = jit(lambda l: grad(curve_length_pure)(l))
        super().__init__(depends_on=[curve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return curve_length_pure(self.curve.incremental_arclength())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """

        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.dJ_dl(self.curve.incremental_arclength()))

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
    """
    Compute the Lp penalty for curvature exceeding a threshold along a curve.

    Args:
        kappa (array-like): Curvature values along the curve.
        gammadash (array-like): Tangent vectors along the curve.
        p (float): The Lp norm exponent.
        desired_kappa (float): The threshold curvature value.

    Returns:
        float: The Lp penalty value.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(kappa-desired_kappa, 0)**p * arc_length)


class LpCurveCurvature(Optimizable):
    r"""
    This class computes a penalty term based on the :math:`L_p` norm
    of the curve's curvature, and penalizes where the local curve curvature exceeds a threshold

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \text{max}(\kappa - \kappa_0, 0)^p ~dl

    where :math:`\kappa_0` is a threshold curvature, given by the argument ``threshold``.

    ``oversample`` (default 1, i.e. no change) evaluates ``kappa()`` on a
    proxy sampled at ``oversample`` times as many quadrature points as
    ``curve`` itself, sharing the same Fourier dofs -- ``curve``'s own fixed
    sampling can under-report the true maximum curvature between its
    points; see :func:`_dense_curve_proxy`.
    """

    def __init__(self, curve, p, threshold=0.0, oversample=1):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda kappa, gammadash: Lp_curvature_pure(kappa, gammadash, p, threshold))
        self.dJ_dkappa = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=0)(kappa, gammadash))
        self.dJ_dgammadash = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=1)(kappa, gammadash))
        self.oversample = int(oversample)
        self._eval_curve, self._eval_base = (
            _dense_curve_proxy(curve, self.oversample) if self.oversample > 1 else (None, None)
        )

    def eval_curve(self):
        if self._eval_curve is None:
            return self.curve
        _resync_dense_curve_proxy(self.curve, self._eval_base)
        return self._eval_curve

    def J(self):
        """
        This returns the value of the quantity.
        """
        c = self.eval_curve()
        return self.J_jax(c.kappa(), c.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        c = self.eval_curve()
        grad0 = self.dJ_dkappa(c.kappa(), c.gammadash())
        grad1 = self.dJ_dgammadash(c.kappa(), c.gammadash())
        raw = c.dkappa_by_dcoeff_vjp(grad0) + c.dgammadash_by_dcoeff_vjp(grad1)
        return _rekey_derivative(raw, c, self.curve)

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def Lp_torsion_pure(torsion, gammadash, p, threshold):
    """
    Compute the Lp penalty for torsion exceeding a threshold along a curve.

    Args:
        torsion (array-like): Torsion values along the curve.
        gammadash (array-like): Tangent vectors along the curve.
        p (float): The Lp norm exponent.
        threshold (float): The threshold torsion value.

    Returns:
        float: The Lp penalty value.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(jnp.abs(torsion)-threshold, 0)**p * arc_length)


class LpCurveTorsion(Optimizable):
    r"""
    LpCurveTorsion is a class that computes a penalty term based on the :math:`L_p` norm
    of the curve's torsion:

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \max(|\tau|-\tau_0, 0)^p ~dl.

    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda torsion, gammadash: Lp_torsion_pure(torsion, gammadash, p, threshold))
        self.dJ_dtorsion = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=0)(torsion, gammadash))
        self.dJ_dgammadash = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=1)(torsion, gammadash))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.torsion(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.dJ_dtorsion(self.curve.torsion(), self.curve.gammadash())
        grad1 = self.dJ_dgammadash(self.curve.torsion(), self.curve.gammadash())
        return self.curve.dtorsion_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}


def cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance, downsample=1):
    """
    Compute the curve-curve distance penalty between two curves.

    Args:
        gamma1 (array-like): Points along the first curve.
        l1 (array-like): Tangent vectors along the first curve.
        gamma2 (array-like): Points along the second curve.
        l2 (array-like): Tangent vectors along the second curve.
        minimum_distance (float): The minimum allowed distance between curves.
        downsample (int, default=1): 
            Factor by which to downsample the quadrature points 
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :]. 
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the 
            total number of quadrature points (since this will produce a nonuniform set of points). 
            This parameter is used to speed up expensive calculations during optimization, 
            while retaining higher accuracy for the other objectives. 

    Returns:
        float: The curve-curve distance penalty value.
    """
    gamma1 = gamma1[::downsample, :]
    gamma2 = gamma2[::downsample, :]
    l1 = l1[::downsample, :]
    l2 = l2[::downsample, :]
    dists = jnp.sqrt(jnp.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
    alen = jnp.linalg.norm(l1, axis=1)[:, None] * jnp.linalg.norm(l2, axis=1)[None, :]
    return jnp.sum(alen * jnp.maximum(minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])


class CurveCurveDistance(Optimizable):
    r"""
    CurveCurveDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} \sum_{j = 1}^{i-1} d_{i,j}

    where 

    .. math::
        d_{i,j} = \int_{\text{curve}_i} \int_{\text{curve}_j} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{r}_j \|_2)^2 ~dl_j ~dl_i\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` are points on coils :math:`i` and :math:`j`, respectively.
    :math:`d_\min` is a desired threshold minimum intercoil distance.  This penalty term is zero when the points on coil :math:`i` and 
    coil :math:`j` lie more than :math:`d_\min` away from one another, for :math:`i, j \in \{1, \cdots, \text{num_coils}\}`

    If num_basecurves is passed, then the code only computes the distance to
    the first `num_basecurves` many curves, which is useful when the coils
    satisfy symmetries that can be exploited.

    NOTE: an ``oversample`` option analogous to :class:`CurveSurfaceDistance`
    and :class:`LpCurveCurvature` was prototyped for this class (evaluating
    each curve on a denser proxy sharing its dofs, to catch a coil-coil
    close approach that falls between native quadrature points) and then
    deliberately *not* shipped: Taylor-testing it against real coil pairs
    showed the same fixed, seeded test producing different, sometimes wildly
    wrong (tens of thousands of percent relative error) gradients on
    different runs -- a non-deterministic bug, not just numerical noise, and
    its root cause (candidate-search related, but not pinned down) was not
    found. See the ``penalty-resolution-patches`` Jarvis diagnostic for the
    full record. Do not re-add this without resolving that first.
    """

    def __init__(self, curves, minimum_distance, num_basecurves=None, downsample=1):
        self.curves = curves
        self.minimum_distance = minimum_distance
        self.downsample = downsample
        args = {"static_argnums": (4,)}
        self.J_jax = jit(lambda gamma1, l1, gamma2, l2, dsample: cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance, dsample), **args)
        self.dJ_dgamma1 = jit(lambda gamma1, l1, gamma2, l2, dsample: grad(self.J_jax, argnums=0)(gamma1, l1, gamma2, l2, dsample), **args)
        self.dJ_dl1 = jit(lambda gamma1, l1, gamma2, l2, dsample: grad(self.J_jax, argnums=1)(gamma1, l1, gamma2, l2, dsample), **args)
        self.dJ_dgamma2 = jit(lambda gamma1, l1, gamma2, l2, dsample: grad(self.J_jax, argnums=2)(gamma1, l1, gamma2, l2, dsample), **args)
        self.dJ_dl2 = jit(lambda gamma1, l1, gamma2, l2, dsample: grad(self.J_jax, argnums=3)(gamma1, l1, gamma2, l2, dsample), **args)
        self.candidates = None
        self.num_basecurves = num_basecurves or len(curves)
        # (image curves of a DipoleArrayField with free shapes depend on their base curves)
        from ..field.dipolearray import curve_dependencies
        super().__init__(depends_on=curve_dependencies(curves))

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(
                [c.gamma()[::self.downsample, :] for c in self.curves], self.minimum_distance, self.num_basecurves)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma()[::self.downsample, :],
                                                           self.curves[j].gamma()[::self.downsample, :])) for i, j in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        return min([np.min(cdist(self.curves[i].gamma()[::self.downsample, :],
                                 self.curves[j].gamma()[::self.downsample, :])) for i in range(len(self.curves)) for j in range(i)])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            res += self.J_jax(gamma1, l1, gamma2, l2, self.downsample)

        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]

        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.dJ_dgamma1(gamma1, l1, gamma2, l2, self.downsample)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.dJ_dl1(gamma1, l1, gamma2, l2, self.downsample)
            dgamma_by_dcoeff_vjp_vecs[j] += self.dJ_dgamma2(gamma1, l1, gamma2, l2, self.downsample)
            dgammadash_by_dcoeff_vjp_vecs[j] += self.dJ_dl2(gamma1, l1, gamma2, l2, self.downsample)

        from ..field.dipolearray import ImageCurve, sum_curve_vjps
        if any(isinstance(c, ImageCurve) for c in self.curves):
            # symmetry images of base curves: only curves in candidate pairs carry a cotangent; one vjp per base curve
            active = sorted({k for pair in self.candidates for k in pair})
            return sum_curve_vjps(self.curves, dgamma_by_dcoeff_vjp_vecs, dgammadash_by_dcoeff_vjp_vecs, active=active)
        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return sum(res)

    return_fn_map = {'J': J, 'dJ': dJ}


def cs_distance_pure(gammac, lc, gammas, ns, minimum_distance):
    """
    Compute the curve-surface distance penalty between a curve and a surface.

    Args:
        gammac (array-like): Points along the curve.
        lc (array-like): Tangent vectors along the curve.
        gammas (array-like): Points on the surface.
        ns (array-like): Surface normal vectors.
        minimum_distance (float): The minimum allowed distance between curve and surface.

    Returns:
        float: The curve-surface distance penalty value.
    """
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None] \
        * jnp.linalg.norm(ns, axis=1)[None, :]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance-dists, 0)**2)


class CurveSurfaceDistance(Optimizable):
    r"""
    CurveSurfaceDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} d_{i}

    where

    .. math::
        d_{i} = \int_{\text{curve}_i} \int_{surface} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{s} \|_2)^2 ~dl_i ~ds\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{s}` are points on coil :math:`i`
    and the surface, respectively. :math:`d_\min` is a desired threshold
    minimum coil-to-surface distance.  This penalty term is zero when the
    points on all coils :math:`i` and on the surface lie more than
    :math:`d_\min` away from one another.

    If ``surface`` is a :class:`BoozerSurface`'s own surface -- i.e. it is
    re-solved on every coil perturbation rather than held fixed -- pass that
    ``BoozerSurface`` as ``boozer_surface`` too. Without it, ``dJ()`` only
    differentiates through the coils and silently omits the term from the
    surface itself moving in response to the same coil perturbation (this
    penalty's ordinary use case is a fixed target plasma boundary, where
    that omitted term is genuinely zero, which is why it was missing here).
    With it, ``dJ()`` adds the missing term via the same implicit-function-
    theorem adjoint that :class:`Iotas` and :class:`MajorRadius` already use
    against the identical cached ``boozer_surface.res['PLU']``/``['vjp']``.

    ``surface_oversample`` (default 1, i.e. no change) evaluates this
    penalty's surface side on an auxiliary, more finely sampled surface that
    shares ``surface``'s own degrees of freedom, rather than ``surface``'s
    own (possibly minimal) quadrature grid -- e.g. a ``BoozerExact`` surface
    is solved on exactly its Nyquist-minimal grid (``2*ntor+1`` by
    ``2*mpol+1`` points), which has no margin to catch a coil passing close
    to the surface at a point between those samples. This mirrors the same
    auxiliary-surface pattern :class:`NonQuasiSymmetricRatio` already uses
    for spectrally accurate evaluation, and does not change what ``surface``
    itself is solved against -- only what this penalty checks.

    Without ``boozer_surface``, ``surface`` is a dependency of this penalty (merged from
    hayashiw/simsopt ``whjh/dipole``): if its dofs are free, ``dJ()`` also returns the derivative
    with respect to them, e.g. for a dipole winding surface that is optimized. Fix the surface's
    dofs when it should be held fixed.
    """

    def __init__(self, curves, surface, minimum_distance, boozer_surface=None, surface_oversample=1):
        self.curves = curves
        self.surface = surface
        self.minimum_distance = minimum_distance
        self.boozer_surface = boozer_surface
        self.surface_oversample = int(surface_oversample)
        self._eval_surface_cache = None
        if self.surface_oversample > 1:
            from .surfacexyztensorfourier import SurfaceXYZTensorFourier
            nphi = self.surface_oversample * len(surface.quadpoints_phi)
            ntheta = self.surface_oversample * len(surface.quadpoints_theta)
            self._eval_surface_cache = SurfaceXYZTensorFourier(
                mpol=surface.mpol, ntor=surface.ntor, nfp=surface.nfp, stellsym=surface.stellsym,
                quadpoints_phi=np.linspace(0, 1, nphi, endpoint=False),
                quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
            )

        self.J_jax = jit(lambda gammac, lc, gammas, ns: cs_distance_pure(gammac, lc, gammas, ns, minimum_distance))
        self.dJ_dgamma = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=0)(gammac, lc, gammas, ns))
        self.dJ_dlc = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=1)(gammac, lc, gammas, ns))
        self.dJ_dgammas = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=2)(gammac, lc, gammas, ns))
        self.dJ_dns = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=3)(gammac, lc, gammas, ns))
        self.candidates = None
        # curves that are not Optimizables (e.g. the fixed copies of a DipoleArrayField) have no dofs
        # (image curves of a DipoleArrayField with free shapes depend on their base curves)
        from ..field.dipolearray import curve_dependencies
        depends_on = curve_dependencies(curves) \
            + ([boozer_surface] if boozer_surface is not None else [surface])
        super().__init__(depends_on=depends_on)

    def eval_surface(self):
        """The surface to sample this penalty's surface side against: either
        ``self.surface`` directly, or (if ``surface_oversample`` > 1) an
        auxiliary denser proxy kept in sync with ``self.surface``'s dofs."""
        if self._eval_surface_cache is None:
            return self.surface
        self._eval_surface_cache.x = self.surface.x
        return self._eval_surface_cache

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
                [c.gamma() for c in self.curves], [self.eval_surface().gamma().reshape((-1, 3))], self.minimum_distance)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.eval_surface().gamma().reshape((-1, 3))
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i, _ in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.eval_surface().gamma().reshape((-1, 3))
        return min([np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i in range(len(self.curves))])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        gammas = self.eval_surface().gamma().reshape((-1, 3))
        ns = self.eval_surface().normal().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            res += self.J_jax(gammac, lc, gammas, ns)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]
        gammas = self.eval_surface().gamma().reshape((-1, 3))
        ns = self.eval_surface().normal().reshape((-1, 3))
        dJ_dgammas_vec = np.zeros_like(gammas)
        dJ_dns_vec = np.zeros_like(ns)
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.dJ_dgamma(gammac, lc, gammas, ns)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.dJ_dlc(gammac, lc, gammas, ns)
            dJ_dgammas_vec += self.dJ_dgammas(gammac, lc, gammas, ns)
            dJ_dns_vec += self.dJ_dns(gammac, lc, gammas, ns)
        from ..field.dipolearray import ImageCurve, sum_curve_vjps
        if any(isinstance(c, ImageCurve) for c in self.curves):
            # symmetry images of base curves: only candidates carry a cotangent; one vjp per base curve
            direct = sum_curve_vjps(self.curves, dgamma_by_dcoeff_vjp_vecs, dgammadash_by_dcoeff_vjp_vecs,
                                    active=[i for i, _ in self.candidates])
        else:
            res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
            direct = sum(res)
        if self.boozer_surface is None:
            # the surface is a dependency: add the derivative with respect to its own dofs
            # (the oversampled proxy shares the surface's dof vector)
            surface = self.eval_surface()
            shape = surface.gamma().shape
            direct = direct + Derivative({self.surface: surface.dgamma_by_dcoeff_vjp(dJ_dgammas_vec.reshape(shape))
                                          + surface.dnormal_by_dcoeff_vjp(dJ_dns_vec.reshape(shape))})
            return direct
        return direct - self._boozer_surface_adjoint_correction(dJ_dgammas_vec, dJ_dns_vec)

    def _boozer_surface_adjoint_correction(self, dJ_dgammas_vec, dJ_dns_vec):
        """
        The implicit-function-theorem correction for the surface itself
        moving in response to the same coil perturbation (see the class
        docstring). Mirrors ``Iotas``/``MajorRadius`` in
        ``surfaceobjectives.py``, reusing the Boozer residual's own cached
        LU factorization and coil-dof vector-Jacobian product rather than
        differentiating through the Newton solve directly.
        """
        booz_surf = self.boozer_surface
        if booz_surf.need_to_run_code:
            res = booz_surf.res
            booz_surf.run_code(res['iota'], G=res['G'])
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = booz_surf.res['vjp']

        surface = self.eval_surface()
        dj_ds = (
            surface.dgamma_by_dcoeff_vjp(dJ_dgammas_vec.reshape(surface.gamma().shape))
            + surface.dnormal_by_dcoeff_vjp(dJ_dns_vec.reshape(surface.normal().shape))
        )
        # dJ_diota = dJ_dG = 0 at the end of dJ_ds; the surface-dof block leads.
        dJ_ds = np.zeros(L.shape[0])
        dJ_ds[:dj_ds.size] = dj_ds

        adj = forward_backward(P, L, U, dJ_ds)
        return dconstraint_dcoils_vjp(adj, booz_surf, iota, G)

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def curve_arclengthvariation_pure(l, mat):
    """
    Compute the variance of the average incremental arclengths over intervals.

    Args:
        l (array-like): Incremental arclengths along the curve.
        mat (array-like): Matrix mapping arclengths to intervals.

    Returns:
        float: The variance of the average arclengths over intervals.
    """
    return jnp.var(mat @ l)


class ArclengthVariation(Optimizable):

    def __init__(self, curve, nintervals="full"):
        r"""
        This class penalizes variation of the arclength along a curve.
        The idea of this class is to avoid ill-posedness of curve objectives due to
        non-uniqueness of the underlying parametrization. Essentially we want to
        achieve constant arclength along the curve. Since we can not expect
        perfectly constant arclength along the entire curve, this class has
        some support to relax this notion. Consider a partition of the :math:`[0, 1]`
        interval into intervals :math:`\{I_i\}_{i=1}^L`, and tenote the average incremental arclength
        on interval :math:`I_i` by :math:`\ell_i`. This objective then penalises the variance

        .. math::
            J = \mathrm{Var}(\ell_i)

        it remains to choose the number of intervals :math:`L` that :math:`[0, 1]` is split into.
        If ``nintervals="full"``, then the number of intervals :math:`L` is equal to the number of quadrature
        points of the curve. If ``nintervals="partial"``, then the argument is as follows:

        A curve in 3d space is defined uniquely by an initial point, an initial
        direction, and the arclength, curvature, and torsion along the curve. For a
        :mod:`simsopt.geo.curvexyzfourier.CurveXYZFourier`, the intuition is now as
        follows: assuming that the curve has order :math:`p`, that means we have
        :math:`3*(2p+1)` degrees of freedom in total. Assuming that three each are
        required for both the initial position and direction, :math:`6p-3` are left
        over for curvature, torsion, and arclength. We want to fix the arclength,
        so we can afford :math:`2p-1` constraints, which corresponds to
        :math:`L=2p`.

        Finally, the user can also provide an integer value for `nintervals`
        and thus specify the number of intervals directly.
        """
        super().__init__(depends_on=[curve])

        assert nintervals in ["full", "partial"] \
            or (isinstance(nintervals, int) and 0 < nintervals <= curve.gamma().shape[0])
        self.curve = curve
        nquadpoints = len(curve.quadpoints)
        if nintervals == "full":
            nintervals = curve.gamma().shape[0]
        elif nintervals == "partial":
            from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
            if isinstance(curve, CurveXYZFourier) or isinstance(curve, JaxCurveXYZFourier):
                nintervals = 2*curve.order
            else:
                raise RuntimeError("Please provide a value other than `partial` for `nintervals`. We only have a default for `CurveXYZFourier` and `JaxCurveXYZFourier`.")

        self.nintervals = nintervals
        indices = np.floor(np.linspace(0, nquadpoints, nintervals+1, endpoint=True)).astype(int)
        mat = np.zeros((nintervals, nquadpoints))
        for i in range(nintervals):
            mat[i, indices[i]:indices[i+1]] = 1/(indices[i+1]-indices[i])
        self.mat = mat
        self.dJ_dl = jit(lambda l: grad(lambda x: curve_arclengthvariation_pure(x, mat))(l))

    def J(self):
        return float(curve_arclengthvariation_pure(self.curve.incremental_arclength(), self.mat))

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.dJ_dl(self.curve.incremental_arclength()))

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def curve_msc_pure(kappa, gammadash):
    """
    Compute the mean squared curvature objective for a curve.

    Args:
        kappa (array-like): Curvature values along the curve.
        gammadash (array-like): Tangent vectors along the curve.

    Returns:
        float: The mean squared curvature value.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(kappa**2 * arc_length)/jnp.mean(arc_length)


class MeanSquaredCurvature(Optimizable):

    def __init__(self, curve, oversample=1):
        r"""
        Compute the mean of the squared curvature of a curve.

        .. math::
            J = (1/L) \int_{\text{curve}} \kappa^2 ~dl

        where :math:`L` is the curve length, :math:`\ell` is the incremental
        arclength, and :math:`\kappa` is the curvature.

        Args:
            curve: the curve of which the curvature should be computed.
            oversample: (default 1, i.e. no change) evaluate on a proxy
                sampled at ``oversample`` times as many quadrature points as
                ``curve`` itself, sharing the same Fourier dofs -- see
                :func:`_dense_curve_proxy`.
        """
        super().__init__(depends_on=[curve])
        self.curve = curve
        self.dJ_dkappa = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=0)(kappa, gammadash))
        self.dJ_dgammadash = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=1)(kappa, gammadash))
        self.oversample = int(oversample)
        self._eval_curve, self._eval_base = (
            _dense_curve_proxy(curve, self.oversample) if self.oversample > 1 else (None, None)
        )

    def eval_curve(self):
        if self._eval_curve is None:
            return self.curve
        _resync_dense_curve_proxy(self.curve, self._eval_base)
        return self._eval_curve

    def J(self):
        c = self.eval_curve()
        return float(curve_msc_pure(c.kappa(), c.gammadash()))

    @derivative_dec
    def dJ(self):
        c = self.eval_curve()
        grad0 = self.dJ_dkappa(c.kappa(), c.gammadash())
        grad1 = self.dJ_dgammadash(c.kappa(), c.gammadash())
        raw = c.dkappa_by_dcoeff_vjp(grad0) + c.dgammadash_by_dcoeff_vjp(grad1)
        return _rekey_derivative(raw, c, self.curve)


@deprecated("`MinimumDistance` has been deprecated and will be removed. Please use `CurveCurveDistance` instead.")
class MinimumDistance(CurveCurveDistance):
    pass


class LinkingNumber(Optimizable):

    def __init__(self, curves, downsample=1):
        Optimizable.__init__(self, depends_on=curves)
        self.curves = curves
        for curve in curves:
            assert np.mod(len(curve.quadpoints), downsample) == 0, f"Downsample {downsample} does not divide the number of quadpoints {len(curve.quadpoints)}."

        self.downsample = downsample
        self.dphis = np.array([(c.quadpoints[1] - c.quadpoints[0]) * downsample for c in self.curves])

        r"""
        Compute the Gauss linking number of a set of curves, i.e. whether the curves
        are interlocked or not.

        The value is an integer, >= 1 if the curves are interlocked, 0 if not. For each pair
        of curves, the contribution to the linking number is
        
        .. math::
            Link(c_1, c_2) = \frac{1}{4\pi} \left| \oint_{c_1}\oint_{c_2}\frac{\textbf{r}_1 - \textbf{r}_2}{|\textbf{r}_1 - \textbf{r}_2|^3} (d\textbf{r}_1 \times d\textbf{r}_2) \right|
            
        where :math:`c_1` is the first curve, :math:`c_2` is the second curve,
        :math:`\textbf{r}_1` is the position vector along the first curve, and
        :math:`\textbf{r}_2` is the position vector along the second curve.

        Args:
            curves (list of Curve, shape (n_curves)): 
                The set of curves for which the linking number should be computed.
            downsample (int, default=1): 
                Factor by which to downsample the quadrature points 
                by skipping through the array by a factor of ``downsample``,
                e.g. curve.gamma()[::downsample, :]. 
                Setting this parameter to a value larger than 1 will speed up the calculation,
                which may be useful if the set of coils is large, though it may introduce
                inaccuracy if ``downsample`` is set too large, or not a multiple of the 
                total number of quadrature points (since this will produce a nonuniform set of points). 
                This parameter is used to speed up expensive calculations during optimization, 
                while retaining higher accuracy for the other objectives. 
        """

    def J(self):
        return sopp.compute_linking_number(
            [c.gamma() for c in self.curves],
            [c.gammadash() for c in self.curves],
            self.dphis,
            self.downsample,
        )

    @derivative_dec
    def dJ(self):
        return Derivative({})
