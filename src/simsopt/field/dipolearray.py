"""A large array of fixed-shape coils (e.g. dipoles) as ONE magnetic-field object.

With ``BiotSavart`` every coil is a ``Coil`` Optimizable (curve + current chain).  For an array of
about 2000 coils whose only free dofs are a few dozen currents, setting the dofs makes simsopt
walk the dependency graph over every coil, which costs tens of seconds per evaluation.
:class:`DipoleArrayField` stores the fixed geometry as plain arrays and the currents as one small
Optimizable (:class:`DipoleArrayCurrents`), so the graph holds a handful of nodes.

Free shapes (2026-11-12): the geometry of selected coils (e.g. the dipoles) can instead be built from a
few *base curves* (real simsopt curves with free dofs).  Every such coil is a symmetry image
``gamma_i = gamma_base(j_i) @ R_i`` of one base curve (``R_i`` a 3x3 rotation, composed from the
``RotatedCurve`` wrappers of the original coil), so only the base curves carry shape dofs and the dependency
graph holds currents + base curves, not thousands of coils.
"""
import numpy as np

import simsoptpp as sopp
from .._core.derivative import Derivative
from .._core.json import GSONDecoder
from .._core.optimizable import Optimizable
from .magneticfield import MagneticField

__all__ = ['DipoleArrayCurrents', 'DipoleArrayField', 'FrozenCurve', 'ImageCurve', 'detect_images',
           'sum_curve_vjps', 'curve_dependencies', 'batched_base_vjps']


class DipoleArrayCurrents(Optimizable):
    """The independent currents (dofs) of a :class:`DipoleArrayField`, in amps.

    Args:
        x0: current values (one per independent current).
        dofs: a ``DOFs`` object (used when loading from JSON; carries names, free flags and bounds).
        kwargs: forwarded to :class:`Optimizable` (``names``, ``fixed``, ``lower_bounds``, ``upper_bounds``).
    """

    def __init__(self, x0, dofs=None, **kwargs):
        if dofs is None:
            Optimizable.__init__(self, x0=np.asarray(x0, dtype=float), **kwargs)
        else:
            Optimizable.__init__(self, dofs=dofs)

    @property
    def x0(self):
        return np.array(self.local_full_x)


class FrozenCurve:
    """A curve with fixed geometry that is not an Optimizable (no dofs, not in any dependency graph).

    Provides what the penalty objectives need (``gamma``, ``gammadash`` and the zero derivative
    vjps).  ``is_dipole`` tells dipole curves from other coils (e.g. the TF coils).
    """

    def __init__(self, gamma, gammadash, is_dipole=False):
        self._gamma = gamma
        self._gammadash = gammadash
        self.is_dipole = bool(is_dipole)
        self.quadpoints = np.linspace(0, 1, len(gamma), endpoint=False)

    def gamma(self):
        return self._gamma

    def gammadash(self):
        return self._gammadash

    def num_dofs(self):
        return 0

    @property
    def x(self):
        return np.zeros(0)

    def dgamma_by_dcoeff_vjp(self, v):
        return Derivative({})

    def dgammadash_by_dcoeff_vjp(self, v):
        return Derivative({})


class ImageCurve:
    """Symmetry image ``gamma = base.gamma() @ rot`` of a base curve (not an Optimizable).

    Used as the ``curves`` entries of a free-shape :class:`DipoleArrayField`: values follow the base curve
    live, the derivative vjps are those of the base curve with the cotangent rotated back by ``rot.T``
    (so they are correct, not zero).  ``dependencies`` lists the Optimizables a penalty has to depend on.
    """

    def __init__(self, base, rot, is_dipole=True):
        self.base = base
        self.rot = np.asarray(rot, dtype=float)
        self.is_dipole = bool(is_dipole)

    @property
    def dependencies(self):
        return [self.base]

    @property
    def quadpoints(self):
        return self.base.quadpoints

    def gamma(self):
        return self.base.gamma() @ self.rot

    def gammadash(self):
        return self.base.gammadash() @ self.rot

    def gammadashdash(self):
        return self.base.gammadashdash() @ self.rot

    def num_dofs(self):
        return self.base.num_dofs()

    @property
    def x(self):
        return self.base.x

    def dgamma_by_dcoeff_vjp(self, v):
        return self.base.dgamma_by_dcoeff_vjp(np.asarray(v) @ self.rot.T)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.base.dgammadash_by_dcoeff_vjp(np.asarray(v) @ self.rot.T)


def curve_dependencies(curves):
    """The Optimizables (unique) that a list of curves, some of them non-Optimizable views, depends on."""
    out, seen = [], set()
    for c in curves:
        for o in ([c] if isinstance(c, Optimizable) else getattr(c, 'dependencies', [])):
            if id(o) not in seen:
                seen.add(id(o))
                out.append(o)
    return out


def _has_free(opt):
    return any(o._dofs._free.any() for o in opt.unique_dof_lineage)


_BATCH_VJP = None


def _batch_vjp_fn():
    """Jitted, vmapped (over base curves) vjp of the superellipse gamma and gammadash w.r.t. its 10 dofs."""
    global _BATCH_VJP
    if _BATCH_VJP is None:
        import jax
        import jax.numpy as jnp
        from ..geo.curvesuperellipse import curve_superellipse_pure

        def one(d, wg, wd, q):
            gam = lambda x: curve_superellipse_pure(x, q)
            gamd = lambda x: jax.jvp(lambda p: curve_superellipse_pure(x, p), (q,), (jnp.ones_like(q),))[1]
            return jax.vjp(gam, d)[1](wg)[0] + jax.vjp(gamd, d)[1](wd)[0]
        _BATCH_VJP = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, None)))
    return _BATCH_VJP


def _batchable(bases):
    """True when all ``bases`` are CurvePlanarOnSurface(ScaledCurveSuperEllipse) with the same quadpoints and a fixed surface."""
    from ..geo import CurvePlanarOnSurface
    from ..geo.curvesuperellipse import ScaledCurveSuperEllipse
    if len(bases) < 2:
        return False
    q0 = None
    for b in bases:
        if not isinstance(b, CurvePlanarOnSurface) or not isinstance(b.curve, ScaledCurveSuperEllipse):
            return False
        if _has_free(b.surface):
            return False
        q = np.asarray(b.curve.curve.quadpoints)
        if q0 is None:
            q0 = q
        elif q.shape != q0.shape or not np.array_equal(q, q0):
            return False
    return True


def batched_base_vjps(bases, rgs, rds):
    """Per base curve ``dgamma_by_dcoeff_vjp(rg) + dgammadash_by_dcoeff_vjp(rd)`` in ONE jitted vmapped JAX call.

    ``bases`` are CurvePlanarOnSurface curves wrapping a ScaledCurveSuperEllipse (identical quadpoints, fixed
    winding surface); ``rgs``/``rds`` the cotangents ``(nbase, nq, 3)``.  Returns the list of ``Derivative``s
    (same dof slots as the per-curve vjps; the fixed surface contributes nothing) or None when the bases are
    not batchable (the caller then loops over the base curves).
    """
    if not _batchable(bases):
        return None
    import jax.numpy as jnp
    F, BR, dofs = [], [], []
    for b in bases:
        b.curve._update_curve()
        n = np.asarray(b.surface.unitnormal()[b.iphi, b.itheta])
        t = np.asarray(b.surface.gammadash2()[b.iphi, b.itheta] if b.align == "theta"
                       else b.surface.gammadash1()[b.iphi, b.itheta])
        e3 = n / np.linalg.norm(n)
        t = t - (t @ e3) * e3
        e1 = t / np.linalg.norm(t)
        F.append(np.stack((e1, np.cross(e3, e1), e3), axis=-1))
        BR.append(b.base_rotation)
        dofs.append(b.curve.curve.get_dofs())
    F, BR, dofs = np.array(F), np.array(BR), np.array(dofs)
    rgs, rds = np.asarray(rgs), np.asarray(rds)
    # w = (v @ F) @ BR^T, per base curve
    wg = np.einsum('jni,jik,jlk->jnl', rgs, F, BR)
    wd = np.einsum('jni,jik,jlk->jnl', rds, F, BR)
    q = jnp.asarray(np.asarray(bases[0].curve.curve.quadpoints))
    out = np.asarray(_batch_vjp_fn()(jnp.asarray(dofs), jnp.asarray(wg), jnp.asarray(wd), q))
    return [b.curve._split_vjp(out[j]) for j, b in enumerate(bases)]


def sum_curve_vjps(curves, vgamma, vgammadash, active=None):
    """``sum_i dgamma_by_dcoeff_vjp(vgamma[i]) + dgammadash_by_dcoeff_vjp(vgammadash[i])`` over ``curves[i]``.

    :class:`ImageCurve` entries are grouped by base curve: the cotangents are rotated back and summed, and
    each base curve with a free dof gets ONE vjp call (instead of one per image).  ``active`` restricts the sum
    to those indices (the others have zero cotangent).
    """
    res = None
    groups = {}
    for i in (range(len(curves)) if active is None else active):
        c = curves[i]
        if isinstance(c, ImageCurve):
            e = groups.get(id(c.base))
            rg, rd = np.asarray(vgamma[i]) @ c.rot.T, np.asarray(vgammadash[i]) @ c.rot.T
            if e is None:
                groups[id(c.base)] = [c.base, rg, rd]
            else:
                e[1] = e[1] + rg
                e[2] = e[2] + rd
        else:
            term = c.dgamma_by_dcoeff_vjp(vgamma[i]) + c.dgammadash_by_dcoeff_vjp(vgammadash[i])
            res = term if res is None else res + term
    todo = [e for e in groups.values() if _has_free(e[0])]
    terms = batched_base_vjps([e[0] for e in todo], [e[1] for e in todo], [e[2] for e in todo]) if todo else []
    if terms is None:
        terms = [base.dgamma_by_dcoeff_vjp(rg) + base.dgammadash_by_dcoeff_vjp(rd) for base, rg, rd in todo]
    for term in terms:
        res = term if res is None else res + term
    return Derivative({}) if res is None else res


def detect_images(coils, is_base=None):
    """Find the base curve and the rotation of every coil.

    Unwraps the :class:`~simsopt.geo.curve.RotatedCurve` wrappers of each coil curve; the innermost curve is
    its root.  Coils whose root satisfies ``is_base`` (default: a :class:`~simsopt.geo.CurvePlanarOnSurface`,
    or a curve with a free dof) become images of that root, all others stay frozen.  Returns
    ``(base_curves, image_base, image_rot)``: the root curves, per coil the index of its base (-1 =
    frozen) and the ``(ncoils, 3, 3)`` matrices with ``gamma_coil = gamma_root @ R`` (composed exactly like
    ``RotatedCurve``, so the quadrature orientation is the same).
    """
    from ..geo.curve import RotatedCurve
    from ..geo import CurvePlanarOnSurface
    if is_base is None:
        def is_base(root):
            return isinstance(root, CurvePlanarOnSurface) or _has_free(root)
    base_curves, where = [], {}
    image_base = -np.ones(len(coils), dtype=int)
    image_rot = np.tile(np.eye(3), (len(coils), 1, 1))
    for i, coil in enumerate(coils):
        chain, curve = [], coil.curve
        while isinstance(curve, RotatedCurve):
            chain.append(curve)
            curve = curve.curve
        if not is_base(curve):
            continue
        if id(curve) not in where:
            where[id(curve)] = len(base_curves)
            base_curves.append(curve)
        image_base[i] = where[id(curve)]
        R = np.eye(3)
        for rc in reversed(chain):  # innermost wrapper first
            R = R @ rc.rotmat
        image_rot[i] = R
    return base_curves, image_base, image_rot


class _FrozenCurrentDofs:
    def __init__(self, field, i):
        self._field, self._i = field, i

    def all_fixed(self):
        return not self._field._coil_has_free_dof(self._i)


class _FrozenCurrent:
    """Read-only view of the (live) current of coil ``i`` of a :class:`DipoleArrayField`."""

    def __init__(self, field, i):
        self._field, self._i = field, i
        self.dofs = _FrozenCurrentDofs(field, i)

    def get_value(self):
        return float(self._field.coil_currents()[self._i])


class _FrozenCoil:
    def __init__(self, curve, current):
        self.curve = curve
        self.current = current


def _factorize_images(gammas, gammadashs, rtol=1e-12):
    """Write every curve as ``root_gamma @ R`` (and ``root_gammadash @ R``) for a few root curves.

    Returns (roots_gamma, roots_gammadash, root_index, R); the images of a symmetric coil set
    are rotated copies of a few base curves, which makes the JSON file small.  A curve that is
    no copy of an earlier root becomes a new root (R = identity).
    """
    roots_g, roots_d, index, rots = [], [], [], []
    for g, d in zip(gammas, gammadashs):
        found = None
        scale = max(np.max(np.abs(g)), 1e-300)
        for r, (g0, d0) in enumerate(zip(roots_g, roots_d)):
            if g0.shape != g.shape:
                continue
            A, Bm = np.concatenate([g0, d0]), np.concatenate([g, d])
            # orthogonal Procrustes (rotations / reflections), accurate to round-off; least squares as fallback
            U, _, Vt = np.linalg.svd(A.T @ Bm)
            R = U @ Vt
            if np.max(np.abs(g0 @ R - g)) > rtol * scale:
                R = np.linalg.lstsq(A, Bm, rcond=None)[0]
            if np.max(np.abs(g0 @ R - g)) <= rtol * scale and np.max(np.abs(d0 @ R - d)) <= rtol * np.max(np.abs(d)):
                found = (r, R)
                break
        if found is None:
            roots_g.append(g)
            roots_d.append(d)
            found = (len(roots_g) - 1, np.eye(3))
        index.append(found[0])
        rots.append(found[1])
    return roots_g, roots_d, np.asarray(index, dtype=int), np.asarray(rots)


class DipoleArrayField(MagneticField):
    r"""Biot-Savart field of many coils with currents linear in a few dofs and (optionally) free base shapes.

    The current of coil :math:`i` is :math:`I_i = \sum_k M_{ik} x_k + c_i` with
    :math:`x` the dofs of ``currents`` (:class:`DipoleArrayCurrents`).  The geometry (``gamma``,
    ``gammadash`` at the quadrature points of each coil) is either

    * frozen: plain arrays (``image_base[i] = -1``), or
    * a symmetry image of a *base curve* (``image_base[i] = j``): ``gamma_i = gamma_{base j} @ R_i``.  The
      base curves are real simsopt curves (parents of the field in the dependency graph) whose free dofs are
      dofs of the field.  When a base curve changes only the unit fields of its images are recomputed;
      when only the currents change the cached unit fields are reused.

    ``B``, ``dB_by_dX`` and ``d2B_by_dXdX`` use simsopt's Biot-Savart kernel on all coils at once
    (identical to :class:`~simsopt.field.biotsavart.BiotSavart`).  ``B_vjp`` and ``B_and_dB_vjp`` return the
    derivative with respect to the currents and, for base curves with free dofs, with respect to their shape
    dofs (per-coil derivative of the geometry from the C++ kernel, rotated back by ``R_i^T``, summed over the
    images of each base curve, one ``dgamma_by_dcoeff_vjp`` per base curve).

    Args:
        currents: a :class:`DipoleArrayCurrents` (shared between views of the same array, see
            :meth:`clone`).
        gammas, gammadashs: per coil, arrays ``(nquad, 3)`` (a list or an array ``(ncoils, nquad, 3)``);
            entries of image coils may be ``None``.
        current_matrix: ``(ncoils, ndofs)`` matrix ``M``.
        current_offset: optional ``(ncoils,)`` vector ``c`` (constant currents).
        is_dipole: optional boolean per coil (dipole vs other coil, for the curve-surface penalty).
        base_curves, image_base, image_rot: optional free-shape data, see :func:`detect_images`.
    """

    def __init__(self, currents, gammas, gammadashs, current_matrix, current_offset=None, is_dipole=None,
                 base_curves=None, image_base=None, image_rot=None):
        self.currents = currents
        n = len(gammas)
        assert len(gammadashs) == n
        self._base_curves = list(base_curves) if base_curves else []
        if self._base_curves:
            self._image_base = np.asarray(image_base, dtype=int)
            self._image_rot = np.ascontiguousarray(image_rot, dtype=float)
        else:
            self._image_base = -np.ones(n, dtype=int)
            self._image_rot = np.tile(np.eye(3), (n, 1, 1))
        assert len(self._image_base) == n and self._image_rot.shape == (n, 3, 3)
        self._base_members = [np.nonzero(self._image_base == j)[0] for j in range(len(self._base_curves))]
        self._gammas = [None if g is None else np.ascontiguousarray(g, dtype=float) for g in gammas]
        self._gammadashs = [None if g is None else np.ascontiguousarray(g, dtype=float) for g in gammadashs]
        self._M = np.ascontiguousarray(current_matrix, dtype=float)
        assert self._M.shape == (n, currents.local_full_dof_size)
        self._c = np.zeros(n) if current_offset is None else np.asarray(current_offset, dtype=float)
        self._is_dipole = np.zeros(n, dtype=bool) if is_dipole is None else np.asarray(is_dipole, dtype=bool)
        self._factorization = None
        self._curves = None
        self._coils_cache = None
        # per-coil unit-current fields at the cached points
        self._cache_pts = None
        self._cache_level = -1
        self._Bi = self._dBi = self._ddBi = None
        self._stale = np.zeros(n, dtype=bool)  # image coils whose cached unit field is out of date
        self._geom_check = True
        self._base_g = [None] * len(self._base_curves)
        self._base_d = [None] * len(self._base_curves)
        MagneticField.__init__(self, depends_on=[currents] + self._base_curves)
        self._sync_geometry()
        assert all(g is not None for g in self._gammas)

    # ---- sizes / currents -------------------------------------------------------------------
    @property
    def ncoils(self):
        return len(self._gammas)

    @property
    def has_free_shapes(self):
        """True if the geometry of some coils is built from base curves (they may have free dofs)."""
        return len(self._base_curves) > 0

    @property
    def base_curves(self):
        return list(self._base_curves)

    @property
    def image_base(self):
        return self._image_base.copy()

    def coil_currents(self):
        """The current of every coil (amps) at the present dofs."""
        return self._M @ np.asarray(self.currents.local_full_x) + self._c

    def _coil_has_free_dof(self, i):
        free = np.asarray(self.currents.local_dofs_free_status)
        if bool(np.any(free & (self._M[i] != 0))):
            return True
        j = self._image_base[i]
        return j >= 0 and _has_free(self._base_curves[j])

    # ---- geometry of the base curves -----------------------------------------------------------
    def _sync_geometry(self):
        """Refresh the gamma / gammadash of the images of every base curve that changed; flag their unit fields stale."""
        if not self._geom_check:
            return
        self._geom_check = False
        for j, curve in enumerate(self._base_curves):
            g, d = curve.gamma(), curve.gammadash()
            if self._base_g[j] is not None and np.array_equal(g, self._base_g[j]) and np.array_equal(d, self._base_d[j]):
                continue
            self._base_g[j], self._base_d[j] = np.array(g), np.array(d)
            idx = self._base_members[j]
            for i in idx:
                self._gammas[i] = np.ascontiguousarray(g @ self._image_rot[i])
                self._gammadashs[i] = np.ascontiguousarray(d @ self._image_rot[i])
            self._stale[idx] = True

    # ---- views of the geometry ---------------------------------------------------------------
    @property
    def curves(self):
        """Coil curves outside the dependency graph: :class:`FrozenCurve` (fixed arrays) or, for coils
        built from a base curve, :class:`ImageCurve` (live; correct derivative vjps)."""
        if self._curves is None:
            self._curves = [
                FrozenCurve(self._gammas[i], self._gammadashs[i], self._is_dipole[i]) if self._image_base[i] < 0
                else ImageCurve(self._base_curves[self._image_base[i]], self._image_rot[i], self._is_dipole[i])
                for i in range(self.ncoils)]
        return self._curves

    @property
    def coils(self):
        """Light read-only coil views (``.curve`` see :attr:`curves`, ``.current.get_value()`` live)."""
        if self._coils_cache is None:
            self._coils_cache = [_FrozenCoil(c, _FrozenCurrent(self, i)) for i, c in enumerate(self.curves)]
        return self._coils_cache

    @property
    def _coils(self):
        return self.coils

    def clone(self):
        """A second field object over the same geometry and the same (shared) currents and base curves.

        It has its own evaluation points and caches, like ``BiotSavart(bs.coils)`` of the original
        implementation, and follows the dofs of ``self.currents`` (and of the base curves).
        """
        new = DipoleArrayField(self.currents, self._gammas, self._gammadashs, self._M, self._c, self._is_dipole,
                               base_curves=self._base_curves, image_base=self._image_base, image_rot=self._image_rot)
        new._factorization = self._factorization
        return new

    # ---- field evaluation ----------------------------------------------------------------------
    def recompute_bell(self, parent=None):
        self.clear_cached_properties()
        if self._base_curves and parent is not self.currents:
            self._geom_check = True

    def _unit_fields(self, idx, pts, derivatives):
        """(Re)compute the unit-current fields of the coils ``idx`` at ``pts``."""
        idx = list(idx)
        for arr, lev in ((self._Bi, 0), (self._dBi, 1), (self._ddBi, 2)):
            if arr is not None and lev <= derivatives:
                arr[idx] = 0.
        Bs = [self._Bi[i] for i in idx]
        dBs = [self._dBi[i] for i in idx] if derivatives >= 1 else []
        ddBs = [self._ddBi[i] for i in idx] if derivatives >= 2 else []
        sopp.biot_savart(np.ascontiguousarray(pts), [self._gammas[i] for i in idx],
                         [self._gammadashs[i] for i in idx], Bs, dBs, ddBs)

    def compute(self, derivatives=0):
        """Evaluate the unit-current field of every coil (and its derivatives) at the points."""
        assert derivatives in (0, 1, 2)
        self._sync_geometry()
        pts = self.get_points_cart_ref()
        npts = len(pts)
        if (self._cache_pts is not None and self._cache_level >= derivatives
                and self._cache_pts.shape == pts.shape and np.array_equal(self._cache_pts, pts)):
            if self._stale.any():  # only some base curves changed: recompute their images
                self._unit_fields(np.nonzero(self._stale)[0], pts, self._cache_level)
                self._stale[:] = False
            return
        n = self.ncoils
        if self._Bi is None or self._Bi.shape[1] != npts:
            self._Bi = np.zeros((n, npts, 3))
            self._dBi = self._ddBi = None
        if derivatives >= 1 and (self._dBi is None or self._dBi.shape[1] != npts):
            self._dBi = np.zeros((n, npts, 3, 3))
        if derivatives >= 2 and (self._ddBi is None or self._ddBi.shape[1] != npts):
            self._ddBi = np.zeros((n, npts, 3, 3, 3))
        self._unit_fields(range(n), pts, derivatives)
        self._stale[:] = False
        self._cache_pts = np.array(pts)
        self._cache_level = derivatives

    def _B_impl(self, B):
        self.compute(0)
        I = self.coil_currents()
        B[:] = (I @ self._Bi.reshape(self.ncoils, -1)).reshape(B.shape)

    def _dB_by_dX_impl(self, dB):
        self.compute(1)
        I = self.coil_currents()
        dB[:] = (I @ self._dBi.reshape(self.ncoils, -1)).reshape(dB.shape)

    def _d2B_by_dXdX_impl(self, ddB):
        self.compute(2)
        I = self.coil_currents()
        ddB[:] = (I @ self._ddBi.reshape(self.ncoils, -1)).reshape(ddB.shape)

    def _A_impl(self, A):
        raise NotImplementedError("DipoleArrayField does not provide the vector potential.")

    _dA_by_dX_impl = _d2A_by_dXdX_impl = _A_impl

    # ---- derivatives with respect to the currents ----------------------------------------------------
    def dB_by_dcoilcurrents(self, compute_derivatives=0):
        self.compute(max(compute_derivatives, 0))
        return list(self._Bi)

    def d2B_by_dXdcoilcurrents(self, compute_derivatives=1):
        self.compute(max(compute_derivatives, 1))
        return list(self._dBi)

    def _shape_vjp(self, v, vgrad=None):
        """Derivative w.r.t. the free base-curve dofs of ``sum v . B`` (and of ``sum vgrad . dB`` if given).

        Returns a ``Derivative`` (or two, for B and grad B), or None when no base curve has a free dof.
        """
        free = [j for j, c in enumerate(self._base_curves) if _has_free(c)]
        if not free:
            return None
        self._sync_geometry()
        coils = np.concatenate([self._base_members[j] for j in free])
        pts = self.get_points_cart_ref()
        gam = [self._gammas[i] for i in coils]
        gamd = [self._gammadashs[i] for i in coils]
        cur = list(self.coil_currents()[coils])
        res_g = [np.zeros_like(g) for g in gam]
        res_d = [np.zeros_like(g) for g in gam]
        v = np.ascontiguousarray(v)
        if vgrad is None:
            sopp.biot_savart_vjp_graph(pts, gam, gamd, cur, v, res_g, res_d, [], [], [])
            sets = [(res_g, res_d)]
        else:
            vgrad = np.ascontiguousarray(vgrad)
            rg2 = [np.zeros_like(g) for g in gam]
            rd2 = [np.zeros_like(g) for g in gam]
            sopp.biot_savart_vjp_graph(pts, gam, gamd, cur, v, res_g, res_d, vgrad, rg2, rd2)
            sets = [(res_g, res_d), (rg2, rd2)]
        out = []
        pos = {int(i): k for k, i in enumerate(coils)}
        for rg, rd in sets:
            total = None
            gjs, djs = [], []
            for j in free:
                rot = self._image_rot[self._base_members[j]]
                ks = [pos[int(i)] for i in self._base_members[j]]
                # gamma_i = gamma_base @ R_i  =>  dJ/dgamma_base = sum_i dJ/dgamma_i @ R_i^T
                gjs.append(sum(rg[k] @ R.T for k, R in zip(ks, rot)))
                djs.append(sum(rd[k] @ R.T for k, R in zip(ks, rot)))
            terms = batched_base_vjps([self._base_curves[j] for j in free], gjs, djs)
            if terms is None:
                terms = [self._base_curves[j].dgamma_by_dcoeff_vjp(g) + self._base_curves[j].dgammadash_by_dcoeff_vjp(d)
                         for j, g, d in zip(free, gjs, djs)]
            for term in terms:
                total = term if total is None else total + term
            out.append(total)
        return out

    def B_vjp(self, v):
        r"""Derivative of :math:`\sum_i v_i\cdot B(x_i)` with respect to the currents and the free base-curve dofs (a ``Derivative``)."""
        self.compute(0)
        res_coil = self._Bi.reshape(self.ncoils, -1) @ np.ascontiguousarray(v).reshape(-1)
        res = Derivative({self.currents: self._M.T @ res_coil})
        sh = self._shape_vjp(v)
        return res if sh is None else res + sh[0]

    def B_and_dB_vjp(self, v, vgrad):
        r"""As :meth:`B_vjp`, returns the pair of derivatives for :math:`B` and :math:`\nabla B`."""
        self.compute(1)
        res_B = self._Bi.reshape(self.ncoils, -1) @ np.ascontiguousarray(v).reshape(-1)
        res_dB = self._dBi.reshape(self.ncoils, -1) @ np.ascontiguousarray(vgrad).reshape(-1)
        a, b = Derivative({self.currents: self._M.T @ res_B}), Derivative({self.currents: self._M.T @ res_dB})
        sh = self._shape_vjp(v, vgrad)
        return (a, b) if sh is None else (a + sh[0], b + sh[1])

    # ---- serialization ------------------------------------------------------------------------------------
    def as_dict(self, serial_objs_dict) -> dict:
        d = {"@module": self.__class__.__module__, "@class": self.__class__.__name__,
             "@name": getattr(self, "name", str(id(self)))}
        try:
            from importlib import import_module
            d["@version"] = str(import_module(self.__class__.__module__.split(".", maxsplit=1)[0]).__version__)
        except (AttributeError, ImportError):
            d["@version"] = None

        def ref(obj):
            name = getattr(obj, "name", str(id(obj)))
            if name not in serial_objs_dict:
                serial_objs_dict[name] = obj.as_dict(serial_objs_dict)
            return {"$type": "ref", "value": name}
        d["currents"] = ref(self.currents)
        # geometry of the frozen coils: a few root curves plus a 3x3 rotation per coil
        frozen = np.nonzero(self._image_base < 0)[0]
        if self._base_curves:
            fact = _factorize_images([self._gammas[i] for i in frozen], [self._gammadashs[i] for i in frozen])
            d["frozen_index"] = frozen
            d["base_curves"] = [ref(c) for c in self._base_curves]
            d["image_base"] = self._image_base
            d["image_rot"] = self._image_rot
        else:
            if self._factorization is None:
                self._factorization = _factorize_images(self._gammas, self._gammadashs)
            fact = self._factorization
        roots_g, roots_d, index, rots = fact
        d["roots_gamma"] = [np.asarray(g) for g in roots_g]
        d["roots_gammadash"] = [np.asarray(g) for g in roots_d]
        d["root_index"] = index
        d["rotations"] = rots
        rows, cols = np.nonzero(self._M)
        d["current_matrix_rows"] = rows
        d["current_matrix_cols"] = cols
        d["current_matrix_vals"] = self._M[rows, cols]
        d["current_offset"] = self._c
        d["is_dipole"] = self._is_dipole
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()

        def dec(key):
            return decoder.process_decoded(d[key], serial_objs_dict=serial_objs_dict, recon_objs=recon_objs)
        currents = dec("currents")
        roots_g, roots_d = dec("roots_gamma"), dec("roots_gammadash")
        index, rots = np.asarray(dec("root_index"), dtype=int), np.asarray(dec("rotations"))
        g_all = [np.asarray(roots_g[r]) @ R for r, R in zip(index, rots)]
        gd_all = [np.asarray(roots_d[r]) @ R for r, R in zip(index, rots)]
        base_curves = image_base = image_rot = None
        if "base_curves" in d:
            base_curves = list(dec("base_curves"))
            image_base = np.asarray(dec("image_base"), dtype=int)
            image_rot = np.asarray(dec("image_rot"))
            n = len(image_base)
            gammas, gammadashs = [None] * n, [None] * n
            for k, i in enumerate(np.asarray(dec("frozen_index"), dtype=int)):
                gammas[i], gammadashs[i] = g_all[k], gd_all[k]
        else:
            gammas, gammadashs = g_all, gd_all
            n = len(gammas)
        M = np.zeros((n, currents.local_full_dof_size))
        M[np.asarray(dec("current_matrix_rows"), dtype=int), np.asarray(dec("current_matrix_cols"), dtype=int)] = \
            np.asarray(dec("current_matrix_vals"))
        field = cls(currents, gammas, gammadashs, M, np.asarray(dec("current_offset")), np.asarray(dec("is_dipole")),
                    base_curves=base_curves, image_base=image_base, image_rot=image_rot)
        if base_curves is None:
            field._factorization = ([np.asarray(g) for g in roots_g], [np.asarray(g) for g in roots_d], index, rots)
        field.set_points_cart(np.ascontiguousarray(dec("points")))
        return field

    @classmethod
    def from_coils(cls, coils, free_shape=False, is_base=None):
        """Build the array field (and its :class:`DipoleArrayCurrents`) from ``Coil`` objects.

        Every coil current must be a linear combination of ``Current`` objects (``Current``,
        ``ScaledCurrent``, ``CurrentSum``).  The independent ``Current`` objects, free or fixed, become
        the dofs, in the order of the lineage of the coil list (free dofs keep the order of
        ``BiotSavart(coils).x``); free flags, bounds and values are copied.  A coil counts as a dipole
        if the root of its curve (below rotations) is a ``CurvePlanarOnSurface``.

        With ``free_shape=False`` (default) the geometry is frozen into arrays (factorized into root
        curves and rotations).  With ``free_shape=True`` every coil whose root curve satisfies ``is_base``
        (default: a ``CurvePlanarOnSurface`` or a curve with free dofs) becomes a symmetry image of that root
        (:func:`detect_images`, no refitting: the rotation is the product of the ``RotatedCurve`` matrices), the
        root curves become parents of the field, and their free dofs are dofs of the field; the other coils
        (e.g. the TF coils) are frozen arrays.
        """
        from .biotsavart import BiotSavart
        from .coil import Current
        from ..geo.curve import RotatedCurve
        from ..geo import CurvePlanarOnSurface
        bs = BiotSavart(coils)
        bases = []
        seen = {}
        for opt in bs.unique_dof_lineage:
            if isinstance(opt, Current):
                seen[id(opt)] = len(bases)
                bases.append(opt)

        def lin(cur):
            """(dict dof -> coefficient, constant) of a current."""
            if id(cur) in seen:
                return {seen[id(cur)]: 1.0}, 0.0
            if hasattr(cur, 'current_to_scale'):
                dct, const = lin(cur.current_to_scale)
                return {k: cur.scale * v for k, v in dct.items()}, cur.scale * const
            if hasattr(cur, 'current_a'):
                da, ca = lin(cur.current_a)
                db, cb = lin(cur.current_b)
                for k, v in db.items():
                    da[k] = da.get(k, 0.) + v
                return da, ca + cb
            raise ValueError(f"Unsupported current type {type(cur).__name__}")

        M = np.zeros((len(coils), len(bases)))
        c = np.zeros(len(coils))
        for i, coil in enumerate(coils):
            dct, const = lin(coil.current)
            for k, v in dct.items():
                M[i, k] += v
            c[i] = const
        x0 = np.array([b.get_value() for b in bases])
        fixed = np.array([not np.any(b.local_dofs_free_status) for b in bases])
        names = [f"I{k}" for k in range(len(bases))]
        lb = np.array([b.local_full_lower_bounds[0] for b in bases])
        ub = np.array([b.local_full_upper_bounds[0] for b in bases])
        currents = DipoleArrayCurrents(x0, names=names, fixed=fixed, lower_bounds=lb, upper_bounds=ub)

        def is_dipole(curve):
            while isinstance(curve, RotatedCurve):
                curve = curve.curve
            return isinstance(curve, CurvePlanarOnSurface)
        dip = [is_dipole(co.curve) for co in coils]
        if free_shape:
            base_curves, image_base, image_rot = detect_images(coils, is_base)
            gam = [None if image_base[i] >= 0 else co.curve.gamma() for i, co in enumerate(coils)]
            gamd = [None if image_base[i] >= 0 else co.curve.gammadash() for i, co in enumerate(coils)]
            return cls(currents, gam, gamd, M, c, dip, base_curves=base_curves, image_base=image_base,
                       image_rot=image_rot)
        # The geometry is stored (and saved to JSON) as rotated copies of a few root curves.  Use exactly
        # these copies as the field's geometry, so that save/load reproduces it bitwise (they differ from
        # the coils' own gamma by round-off, about 1e-14 m).
        gam, gamd = [co.curve.gamma() for co in coils], [co.curve.gammadash() for co in coils]
        fact = _factorize_images(gam, gamd)
        gam = [fact[0][r] @ R for r, R in zip(fact[2], fact[3])]
        gamd = [fact[1][r] @ R for r, R in zip(fact[2], fact[3])]
        field = cls(currents, gam, gamd, M, c, dip)
        field._factorization = fact
        return field
