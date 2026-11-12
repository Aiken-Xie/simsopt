"""A large array of fixed-shape coils (e.g. dipoles) as ONE magnetic-field object.

With ``BiotSavart`` every coil is a ``Coil`` Optimizable (curve + current chain).  For an array of
about 2000 coils whose only free dofs are a few dozen currents, setting the dofs makes simsopt
walk the dependency graph over every coil, which costs tens of seconds per evaluation.
:class:`DipoleArrayField` stores the fixed geometry as plain arrays and the currents as one small
Optimizable (:class:`DipoleArrayCurrents`), so the graph holds a handful of nodes.
"""
import numpy as np

import simsoptpp as sopp
from .._core.derivative import Derivative
from .._core.json import GSONDecoder
from .._core.optimizable import Optimizable
from .magneticfield import MagneticField

__all__ = ['DipoleArrayCurrents', 'DipoleArrayField', 'FrozenCurve']


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
    r"""Biot-Savart field of many coils with fixed geometry and currents linear in a few dofs.

    The current of coil :math:`i` is :math:`I_i = \sum_k M_{ik} x_k + c_i` with
    :math:`x` the dofs of ``currents`` (:class:`DipoleArrayCurrents`).  The geometry (``gamma``,
    ``gammadash`` at the quadrature points of each coil) is stored as plain arrays.  ``B``, ``dB_by_dX``
    and ``d2B_by_dXdX`` use simsopt's Biot-Savart kernel on all coils at once (identical to
    :class:`~simsopt.field.biotsavart.BiotSavart`).  ``B_vjp`` and ``B_and_dB_vjp`` return the
    derivative with respect to the currents only (the geometry is fixed).

    Args:
        currents: a :class:`DipoleArrayCurrents` (shared between views of the same array, see
            :meth:`clone`).
        gammas, gammadashs: per coil, arrays ``(nquad, 3)`` (a list or an array ``(ncoils, nquad, 3)``).
        current_matrix: ``(ncoils, ndofs)`` matrix ``M``.
        current_offset: optional ``(ncoils,)`` vector ``c`` (constant currents).
        is_dipole: optional boolean per coil (dipole vs other coil, for the curve-surface penalty).
    """

    def __init__(self, currents, gammas, gammadashs, current_matrix, current_offset=None, is_dipole=None):
        self.currents = currents
        self._gammas = [np.ascontiguousarray(g, dtype=float) for g in gammas]
        self._gammadashs = [np.ascontiguousarray(g, dtype=float) for g in gammadashs]
        n = len(self._gammas)
        assert len(self._gammadashs) == n
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
        MagneticField.__init__(self, depends_on=[currents])

    # ---- sizes / currents -------------------------------------------------------------------
    @property
    def ncoils(self):
        return len(self._gammas)

    def coil_currents(self):
        """The current of every coil (amps) at the present dofs."""
        return self._M @ np.asarray(self.currents.local_full_x) + self._c

    def _coil_has_free_dof(self, i):
        free = np.asarray(self.currents.local_dofs_free_status)
        return bool(np.any(free & (self._M[i] != 0)))

    # ---- views of the geometry ---------------------------------------------------------------
    @property
    def curves(self):
        """Fixed copies of the coil curves (:class:`FrozenCurve`, outside the dependency graph)."""
        if self._curves is None:
            self._curves = [FrozenCurve(g, d, dip) for g, d, dip in zip(self._gammas, self._gammadashs, self._is_dipole)]
        return self._curves

    @property
    def coils(self):
        """Light read-only coil views (``.curve`` fixed copy, ``.current.get_value()`` live)."""
        if self._coils_cache is None:
            self._coils_cache = [_FrozenCoil(c, _FrozenCurrent(self, i)) for i, c in enumerate(self.curves)]
        return self._coils_cache

    @property
    def _coils(self):
        return self.coils

    def clone(self):
        """A second field object over the same geometry and the same (shared) currents.

        It has its own evaluation points and caches, like ``BiotSavart(bs.coils)`` of the original
        implementation, and follows the dofs of ``self.currents``.
        """
        new = DipoleArrayField(self.currents, self._gammas, self._gammadashs, self._M, self._c, self._is_dipole)
        new._factorization = self._factorization
        new._curves = self._curves
        return new

    # ---- field evaluation ----------------------------------------------------------------------
    def recompute_bell(self, parent=None):
        self.clear_cached_properties()

    def compute(self, derivatives=0):
        """Evaluate the unit-current field of every coil (and its derivatives) at the points."""
        assert derivatives in (0, 1, 2)
        pts = self.get_points_cart_ref()
        npts = len(pts)
        if (self._cache_pts is not None and self._cache_level >= derivatives
                and self._cache_pts.shape == pts.shape and np.array_equal(self._cache_pts, pts)):
            return
        n = self.ncoils
        if self._Bi is None or self._Bi.shape[1] != npts:
            self._Bi = np.zeros((n, npts, 3))
            self._dBi = self._ddBi = None
        if derivatives >= 1 and (self._dBi is None or self._dBi.shape[1] != npts):
            self._dBi = np.zeros((n, npts, 3, 3))
        if derivatives >= 2 and (self._ddBi is None or self._ddBi.shape[1] != npts):
            self._ddBi = np.zeros((n, npts, 3, 3, 3))
        self._Bi.fill(0.)
        Bs = list(self._Bi)
        dBs, ddBs = [], []
        if derivatives >= 1:
            self._dBi.fill(0.)
            dBs = list(self._dBi)
        if derivatives >= 2:
            self._ddBi.fill(0.)
            ddBs = list(self._ddBi)
        sopp.biot_savart(np.ascontiguousarray(pts), self._gammas, self._gammadashs, Bs, dBs, ddBs)
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

    def B_vjp(self, v):
        r"""Derivative of :math:`\sum_i v_i\cdot B(x_i)` with respect to the currents (a ``Derivative``)."""
        self.compute(0)
        res_coil = self._Bi.reshape(self.ncoils, -1) @ np.ascontiguousarray(v).reshape(-1)
        return Derivative({self.currents: self._M.T @ res_coil})

    def B_and_dB_vjp(self, v, vgrad):
        r"""As :meth:`B_vjp`, returns the pair of derivatives for :math:`B` and :math:`\nabla B`."""
        self.compute(1)
        res_B = self._Bi.reshape(self.ncoils, -1) @ np.ascontiguousarray(v).reshape(-1)
        res_dB = self._dBi.reshape(self.ncoils, -1) @ np.ascontiguousarray(vgrad).reshape(-1)
        return (Derivative({self.currents: self._M.T @ res_B}),
                Derivative({self.currents: self._M.T @ res_dB}))

    # ---- serialization ------------------------------------------------------------------------------------
    def as_dict(self, serial_objs_dict) -> dict:
        d = {"@module": self.__class__.__module__, "@class": self.__class__.__name__,
             "@name": getattr(self, "name", str(id(self)))}
        try:
            from importlib import import_module
            d["@version"] = str(import_module(self.__class__.__module__.split(".", maxsplit=1)[0]).__version__)
        except (AttributeError, ImportError):
            d["@version"] = None
        name = getattr(self.currents, "name", str(id(self.currents)))
        if name not in serial_objs_dict:
            serial_objs_dict[name] = self.currents.as_dict(serial_objs_dict)
        d["currents"] = {"$type": "ref", "value": name}
        if self._factorization is None:
            self._factorization = _factorize_images(self._gammas, self._gammadashs)
        roots_g, roots_d, index, rots = self._factorization
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
        gammas = [np.asarray(roots_g[r]) @ R for r, R in zip(index, rots)]
        gammadashs = [np.asarray(roots_d[r]) @ R for r, R in zip(index, rots)]
        n = len(gammas)
        M = np.zeros((n, currents.local_full_dof_size))
        M[np.asarray(dec("current_matrix_rows"), dtype=int), np.asarray(dec("current_matrix_cols"), dtype=int)] = \
            np.asarray(dec("current_matrix_vals"))
        field = cls(currents, gammas, gammadashs, M, np.asarray(dec("current_offset")), np.asarray(dec("is_dipole")))
        field._factorization = ([np.asarray(g) for g in roots_g], [np.asarray(g) for g in roots_d], index, rots)
        field.set_points_cart(np.ascontiguousarray(dec("points")))
        return field

    @classmethod
    def from_coils(cls, coils):
        """Build the array field (and its :class:`DipoleArrayCurrents`) from ``Coil`` objects.

        Every coil current must be a linear combination of ``Current`` objects (``Current``,
        ``ScaledCurrent``, ``CurrentSum``).  The independent ``Current`` objects, free or fixed, become
        the dofs, in the order of the lineage of the coil list (free dofs keep the order of
        ``BiotSavart(coils).x``); free flags, bounds and values are copied.  A coil counts as a dipole
        if the root of its curve (below rotations) is a ``CurvePlanarOnSurface``.
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
        # The geometry is stored (and saved to JSON) as rotated copies of a few root curves.  Use exactly
        # these copies as the field's geometry, so that save/load reproduces it bitwise (they differ from
        # the coils' own gamma by round-off, about 1e-14 m).
        gam, gamd = [co.curve.gamma() for co in coils], [co.curve.gammadash() for co in coils]
        fact = _factorize_images(gam, gamd)
        gam = [fact[0][r] @ R for r, R in zip(fact[2], fact[3])]
        gamd = [fact[1][r] @ R for r, R in zip(fact[2], fact[3])]
        field = cls(currents, gam, gamd, M, c, [is_dipole(co.curve) for co in coils])
        field._factorization = fact
        return field
