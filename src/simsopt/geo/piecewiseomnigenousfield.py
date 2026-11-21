from functools import partial

import numpy as np
from jax import jacfwd
from jax import jit as jaxjit
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from scipy.optimize import minimize

from simsopt._core import Optimizable

__all__ = ["PiecewiseOmnigenousField"]

_TILE_RADIUS = 4
_TILE_SOFTMIN = 1e-3
_SHAPE_FLOOR = 1e-6
_SHAPE_MARGIN = 1e-2
_INITIAL_SHAPE_FRACTION = 5e-2
_EXPONENT_CAP = 1e6


def _inverse_softplus(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Softplus-transformed quantities must be finite and positive")
    return value + np.log(-np.expm1(-value))


def _logit(value):
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("Sigmoid fractions must be finite and strictly between 0 and 1")
    return np.log(value) - np.log1p(-value)


def _raw_amplitudes(Bmax, Bmin):
    Bmax, Bmin = float(Bmax), float(Bmin)
    if not np.isfinite(Bmax) or not np.isfinite(Bmin):
        raise ValueError("Bmax and Bmin must be finite")
    if Bmin <= 0.0:
        raise ValueError("Bmin must be positive")
    if Bmax <= Bmin:
        raise ValueError("Bmax must be greater than Bmin")
    return _inverse_softplus(Bmax - Bmin), _inverse_softplus(Bmin)


def _raw_p(p, pmin):
    p, pmin = float(p), float(pmin)
    if not np.isfinite(p) or p <= pmin:
        raise ValueError(f"p must be finite and greater than pmin={pmin}")
    return _inverse_softplus(p - pmin)


def _shape_limits(nfp, sigma2, iota):
    """Upper limits of the bounded f and k shape fractions."""
    f_hi = 1.0 - _SHAPE_MARGIN
    t2_hi = (1.0 - _SHAPE_MARGIN) * nfp
    k_hi = (jnp.pi/nfp) * t2_hi / (1.0 + sigma2*t2_hi/iota)
    return f_hi, k_hi


def _decode_shape(qf, qk, iota, nfp, sigma2):
    r"""
    Decode the latent shape coordinates into the parallelogram geometry of the selected
    single-wrap branch (``sigma2 = sign(iota)``, ``sigma1 = -sigma2``). With
    :math:`x=|t_1|`, :math:`y=|t_2|`,

    .. math::
        f = n_{fp} x / (1 + x y), \qquad k = (\pi / n_{fp})\, y / (1 + \sigma_2 y / \iota),

    inverted as :math:`y = k n_{fp} / (\pi - k n_{fp} \sigma_2 / \iota)` and
    :math:`x = f / (n_{fp} - f y)`. The sigmoids cap f below 1 and y below :math:`n_{fp}`, so
    :math:`A = 1 - t_1 t_2 > 0` and :math:`D = 1 + t_2/\iota > 0` at every optimizer iterate.
    """
    f_hi, k_hi = _shape_limits(nfp, sigma2, iota)
    f = _SHAPE_FLOOR + (f_hi - _SHAPE_FLOOR)*jnp.exp(-jnp.logaddexp(0.0, -qf))
    k = _SHAPE_FLOOR + (k_hi - _SHAPE_FLOOR)*jnp.exp(-jnp.logaddexp(0.0, -qk))
    y = k*nfp/(jnp.pi - k*nfp*sigma2/iota)
    xabs = f/(nfp - f*y)
    t1 = -sigma2*xabs
    t2 = sigma2*y
    w1 = jnp.pi*(1. - t1*t2)/(nfp*(1. + t2/iota))
    return f, k, f_hi, k_hi, t1, t2, w1


@partial(jaxjit, static_argnames=("nfp", "sigma2", "pmin"))
def _pw_evaluate(x, iota, phi, theta, zeta_c, theta_c, nfp, sigma2, pmin):
    raw_gap, raw_min, qf, qk, praw = x[0], x[1], x[2], x[3], x[4]
    Bmin = jnp.logaddexp(0., raw_min)
    Bmax = Bmin + jnp.logaddexp(0., raw_gap)
    p = pmin + jnp.logaddexp(0., praw)
    _, _, _, _, t1, t2, w1 = _decode_shape(qf, qk, iota, nfp, sigma2)

    period_phi = 2*jnp.pi/nfp
    period_theta = 2*jnp.pi
    shifts = np.arange(-_TILE_RADIUS, _TILE_RADIUS + 1)
    tile_phi, tile_theta = np.meshgrid(shifts, shifts, indexing="ij")
    tile_phi = jnp.asarray(tile_phi.reshape(-1, 1, 1))
    tile_theta = jnp.asarray(tile_theta.reshape(-1, 1, 1))

    # Wrap only the fixed evaluation coordinates, then include the neighbouring translated
    # parallelograms, so the wrap does not enter the derivatives.
    dphi0 = jnp.mod(jnp.asarray(phi) - zeta_c + period_phi/2, period_phi) - period_phi/2
    dtheta0 = jnp.mod(jnp.asarray(theta) - theta_c + jnp.pi, period_theta) - jnp.pi
    dphi = dphi0[None, ...] - tile_phi*period_phi
    dtheta = dtheta0[None, ...] - tile_theta*period_theta
    u = (dphi + t1*dtheta)/w1
    v = (dtheta + t2*dphi)/jnp.pi
    # Powers beyond the cap already underflow exp(-exponent); capping them keeps exploratory
    # optimizer steps near a degenerate width from overflowing.
    log_cap = jnp.log(_EXPONENT_CAP)
    u_power = jnp.exp(jnp.minimum(2*p*jnp.log(jnp.maximum(jnp.abs(u), 1e-14)), log_cap))
    v_power = jnp.exp(jnp.minimum(2*p*jnp.log(jnp.maximum(jnp.abs(v), 1e-14)), log_cap))
    exponent = u_power + v_power

    # A narrow soft minimum selects the closest tile while keeping the derivatives continuous.
    tau = _TILE_SOFTMIN
    tiled_exponent = -tau*logsumexp(-exponent/tau, axis=0)
    profile = jnp.minimum(jnp.exp(-tiled_exponent), 1.)
    return Bmin + (Bmax - Bmin)*profile


def _pw_fit_model(y, iota, phi, theta, nfp, sigma2, pmin):
    return _pw_evaluate(y[:5], iota, phi, theta, y[5], y[6], nfp=nfp, sigma2=sigma2, pmin=pmin)


_pw_dx = jaxjit(jacfwd(_pw_evaluate, argnums=0), static_argnames=("nfp", "sigma2", "pmin"))
_pw_diota = jaxjit(jacfwd(_pw_evaluate, argnums=1), static_argnames=("nfp", "sigma2", "pmin"))
_pw_fit_jac = jaxjit(jacfwd(_pw_fit_model, argnums=0), static_argnames=("nfp", "sigma2", "pmin"))
_pw_fit_eval = jaxjit(_pw_fit_model, static_argnames=("nfp", "sigma2", "pmin"))


class PiecewiseOmnigenousField(Optimizable):
    r"""
    Piecewise-omnigenous (pwO) target field: a periodically tiled super-Gaussian parallelogram

    .. math::
        B = B_{\min} + (B_{\max} - B_{\min}) \exp\!\left(-\left(|u|^{2p} + |v|^{2p}\right)\right)

    in the Boozer angles, repeated over the :math:`(2\pi/n_{fp}, 2\pi)` lattice. A narrow
    soft minimum of the translated exponents selects the closest tile. It exposes the same
    interface as `OmnigenousField` that `OmnigenousResidual` needs (``phi``, ``theta``,
    ``modB``, ``set_iota``, ``dmodB_by_dparams``, ``dmodB_by_diota``, ``fit_to``).

    The dofs ``Bmax, Bmin, f, k, p`` are unconstrained raw coordinates (the "bounded-fk-v1"
    parameterization): ``Bmin = softplus(raw_min)``, ``Bmax = Bmin + softplus(raw_gap)``,
    ``p = pmin + softplus(raw_p)``, and sigmoid-bounded shape fractions ``f``, ``k`` that keep
    the selected single-wrap branch valid at every iterate. The physical values are
    available through `physical_dofs` and set with `set_physical_dofs`.

    The branch is selected by the sign of the initial iota, and iota cannot change sign
    afterwards. The tile centre (``zeta_c``, ``theta_c``) is not a dof: `fit_to` places it at
    the maximum of the fitted field and then freezes it. `lock_iota_reference` freezes the
    physical geometry at one iota, so the target no longer depends on the surface's iota.

    Args:
        Bmax, Bmin: initial extrema (they set the amplitude dofs; ignored if ``dofs`` is given).
        nfp: number of field periods.
        iota: rotational transform that selects the branch.
        phi, theta: grids in radians, or numbers of points over one field period / full turn.
        pmin: lower bound of the super-Gaussian exponent, at least 1 (default 1).
        dofs: used when loading a saved target.
        zeta_c, theta_c: tile centre (default ``pi/nfp``, 0).
        iota_ref: if not None, lock the geometry at this iota.
    """

    pmin_default = 1.0
    parameterization = "bounded-fk-v1"

    def __init__(self, Bmax, Bmin, nfp, iota, phi=128, theta=64, pmin=None, dofs=None,
                 zeta_c=None, theta_c=None, iota_ref=None):
        self._nfp = int(nfp)
        if self._nfp <= 0:
            raise ValueError("nfp must be positive")
        iota = float(iota)
        if not np.isfinite(iota) or iota == 0.0:
            raise ValueError("Piecewise omnigeneity requires finite nonzero iota")
        pmin = self.pmin_default if pmin is None else float(pmin)
        if not np.isfinite(pmin) or pmin < 1.0:
            raise ValueError("Piecewise pmin must be finite and at least 1")
        self._pmin = pmin
        self._iota = iota
        self._iota_ref = None
        # The connected branch is selected once and mirrored for negative iota.
        self._sigma2 = 1.0 if iota > 0.0 else -1.0

        if isinstance(phi, (int, np.integer)):
            phi = np.linspace(0, 2*np.pi/self._nfp, phi, endpoint=False)
        if isinstance(theta, (int, np.integer)):
            theta = np.linspace(0, 2*np.pi, theta, endpoint=False)
        self._phi, self._theta = np.asarray(phi, dtype=float), np.asarray(theta, dtype=float)
        self._phi_grid, self._theta_grid = np.meshgrid(self._phi, self._theta, indexing="ij")
        self._zeta_c = np.pi/self._nfp if zeta_c is None else float(zeta_c)
        self._theta_c = 0.0 if theta_c is None else float(theta_c)

        names = ["Bmax", "Bmin", "f", "k", "p"]
        if dofs is None:
            raw_gap, raw_min = _raw_amplitudes(Bmax, Bmin)
            raw_shape = _logit(_INITIAL_SHAPE_FRACTION)
            x0 = np.array([raw_gap, raw_min, raw_shape, raw_shape,
                           _raw_p(self._pmin + 1.0, self._pmin)])
            Optimizable.__init__(self, x0=x0, names=names)
        else:
            Optimizable.__init__(self, dofs=dofs)
        if iota_ref is not None:
            self._iota_ref = float(iota_ref)
            self._validate_iota(self._iota_ref)
            self._iota = self._iota_ref
        self.need_to_update = True

    # --- basic properties -------------------------------------------------------------
    @property
    def nfp(self):
        return self._nfp

    @property
    def pmin(self):
        return self._pmin

    @property
    def phi(self):
        return self._phi

    @property
    def theta(self):
        return self._theta

    @property
    def nphi(self):
        return self._phi.size

    @property
    def ntheta(self):
        return self._theta.size

    @property
    def iota(self):
        return self._iota

    @property
    def iota_ref(self):
        return self._iota_ref

    @property
    def zeta_c(self):
        return self._zeta_c

    @property
    def theta_c(self):
        return self._theta_c

    @property
    def Bmin(self):
        return float(np.logaddexp(0., self.local_full_x[1]))

    @property
    def Bmax(self):
        x = self.local_full_x
        return float(np.logaddexp(0., x[1]) + np.logaddexp(0., x[0]))

    @property
    def p(self):
        return float(self._pmin + np.logaddexp(0., self.local_full_x[4]))

    @property
    def modB(self):
        if self.need_to_update:
            self.update()
        return self._modB

    @property
    def ndofs(self):
        return 5

    # --- dofs -----------------------------------------------------------------------
    def get_dofs(self):
        return np.array(self.local_full_x, dtype=float)

    def set_dofs(self, new_values):
        new_values = np.asarray(new_values, dtype=float)
        if new_values.shape != (5,):
            raise ValueError("Piecewise field requires 5 DOFs")
        if np.array_equal(np.array(self.local_full_x, dtype=float), new_values):
            return
        self.local_full_x = new_values.copy()
        self.need_to_update = True

    def physical_dofs(self):
        """Physical ``Bmax, Bmin, f, k, p`` rather than the raw dofs."""
        g = self.piecewise_geometry()
        return np.array([self.Bmax, self.Bmin, g["f"], g["k"], self.p])

    def set_physical_dofs(self, values):
        """Encode physical ``Bmax, Bmin, f, k, p`` into the raw dofs."""
        values = np.asarray(values, dtype=float)
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError("Piecewise physical DOFs must be five finite values")
        raw_gap, raw_min = _raw_amplitudes(values[0], values[1])
        f_hi, k_hi = (float(a) for a in _shape_limits(self._nfp, self._sigma2, self._iota))
        sf = (values[2] - _SHAPE_FLOOR)/(f_hi - _SHAPE_FLOOR)
        sk = (values[3] - _SHAPE_FLOOR)/(k_hi - _SHAPE_FLOOR)
        self.set_dofs(np.array([raw_gap, raw_min, _logit(sf), _logit(sk),
                                _raw_p(values[4], self._pmin)]))

    def set_Bmax(self, Bmax):
        x = self.get_dofs()
        x[0], _ = _raw_amplitudes(Bmax, self.Bmin)
        self.set_dofs(x)

    def set_Bmin(self, Bmin):
        x = self.get_dofs()
        Bmax = self.Bmax
        x[0], x[1] = _raw_amplitudes(Bmax, Bmin)
        self.set_dofs(x)

    # --- iota and geometry ------------------------------------------------------------
    def _validate_iota(self, iota):
        iota = float(iota)
        if not np.isfinite(iota) or iota == 0.0:
            raise ValueError("Piecewise omnigeneity requires finite nonzero iota")
        if np.sign(iota) != self._sigma2:
            raise ValueError("iota cannot cross zero in a fixed piecewise winding branch")

    def set_iota(self, iota):
        """Set the iota the target is evaluated at; ignored while the reference is locked."""
        self._validate_iota(iota)
        new = float(iota) if self._iota_ref is None else self._iota_ref
        if new != self._iota:
            self._iota = new
            self.need_to_update = True

    def lock_iota_reference(self, iota=None):
        """Freeze the physical geometry at one reference iota (``dmodB_by_diota`` becomes 0)."""
        iota = self._iota if iota is None else iota
        self._validate_iota(iota)
        self._iota_ref = float(iota)
        self._iota = float(iota)
        self.set_recompute_flag()

    def unlock_iota_reference(self):
        """Let the geometry follow subsequent equilibrium solves again."""
        self._iota_ref = None
        self.set_recompute_flag()

    def piecewise_geometry(self):
        """Live physical geometry of the selected branch (f, k, t1, t2, A, D, w1, w2, p, ...)."""
        self._validate_iota(self._iota)
        x = self.local_full_x
        f, k, f_hi, k_hi, t1, t2, w1 = (
            float(a) for a in _decode_shape(x[2], x[3], self._iota, self._nfp, self._sigma2))
        t2_hi = (1.0 - _SHAPE_MARGIN)*self._nfp
        return {
            "f": f, "k": k, "f_hi": f_hi, "k_hi": k_hi,
            "k_max": float(np.pi*abs(self._iota)/(abs(self._iota) + self._nfp)),
            "t2_hi": float(t2_hi), "t1": t1, "t2": t2,
            "sigma1": -self._sigma2, "sigma2": self._sigma2,
            "A": 1.0 - t1*t2, "D": 1.0 + t2/self._iota,
            "w1": w1, "w2": float(np.pi), "p": self.p,
        }

    def _statics(self):
        return dict(nfp=self._nfp, sigma2=self._sigma2, pmin=self._pmin)

    # --- evaluation -------------------------------------------------------------------
    def _evaluate(self, x, iota, phi, theta, zeta_c=None, theta_c=None):
        return _pw_evaluate(
            jnp.asarray(x), iota, phi, theta,
            self._zeta_c if zeta_c is None else zeta_c,
            self._theta_c if theta_c is None else theta_c, **self._statics())

    def update(self):
        self._validate_iota(self._iota)
        self._modB = np.asarray(self._evaluate(self.local_full_x, self._iota,
                                               self._phi_grid, self._theta_grid))
        self.need_to_update = False

    def dmodB_by_dparams(self):
        r"""Derivative of ``modB`` with respect to the five raw dofs, shape (nphi, ntheta, 5)."""
        d = np.asarray(_pw_dx(jnp.asarray(self.local_full_x), self._iota, self._phi_grid,
                              self._theta_grid, self._zeta_c, self._theta_c, **self._statics()))
        if d.shape != self._phi_grid.shape + (5,):
            raise RuntimeError(f"Unexpected piecewise derivative shape {d.shape}")
        return d

    def dmodB_by_diota(self):
        r"""Derivative of ``modB`` with respect to iota; zero while the reference is locked."""
        if self._iota_ref is not None:
            return np.zeros(self._phi_grid.shape)
        return np.asarray(_pw_diota(jnp.asarray(self.local_full_x), self._iota, self._phi_grid,
                                    self._theta_grid, self._zeta_c, self._theta_c,
                                    **self._statics()))

    def recompute_bell(self, parent=None):
        self.need_to_update = True

    # --- fitting ----------------------------------------------------------------------
    def fit_to(self, coil_modB, **kwargs):
        r"""
        Fit the five dofs to ``coil_modB`` (shape (nphi, ntheta)) with SLSQP, minimizing the
        normalized residual. The tile centre is a nuisance coordinate: it starts at the grid
        point of maximum ``coil_modB`` and is optimized together with the dofs, then reduced to
        one period and frozen (``zeta_c``, ``theta_c``). The best finite iterate is kept.
        """
        from simsopt.util import proc0_print
        self._validate_iota(self._iota)
        coil_modB = np.asarray(coil_modB, dtype=float)
        if coil_modB.shape != self._phi_grid.shape:
            raise ValueError("coil_modB must have the shape (nphi, ntheta) of the target grid")
        rms_modB = np.sqrt(np.mean(coil_modB**2))
        kwargs.pop("slope_lb", None)  # no slope constraint: the piecewise map is bounded by construction
        statics = self._statics()
        iota = self._iota

        max_index = np.unravel_index(np.argmax(coil_modB), coil_modB.shape)
        initial_centers = np.array([self._phi_grid[max_index], self._theta_grid[max_index]],
                                   dtype=float)
        x0 = np.concatenate([self.get_dofs(), initial_centers])

        cache = dict(x=None, r=None, J=None, valid=False)
        tracker = dict(nfev=0)
        best = dict(x=x0.copy(), objective=np.inf)

        def residual(x):
            if (cache["x"] is None) or (not np.array_equal(cache["x"], x)):
                tracker["nfev"] += 1
                cache["x"] = np.asarray(x).copy()
                candidate = np.asarray(_pw_fit_eval(jnp.asarray(x), iota, self._phi_grid,
                                                    self._theta_grid, **statics))
                cache["valid"] = bool(np.all(np.isfinite(candidate)))
                if cache["valid"]:
                    cache["r"] = (coil_modB - candidate).ravel()/rms_modB
                    objective = 0.5*float(cache["r"] @ cache["r"])
                    if objective < best["objective"]:
                        best["x"] = np.asarray(x).copy()
                        best["objective"] = objective
                else:
                    cache["r"] = np.full(coil_modB.size, 1e6)
                cache["J"] = None
                proc0_print(f"nfev={tracker['nfev']}, mean(r^2)={np.mean(cache['r']**2):.6e}")
            return cache["r"]

        def jac(x):
            residual(x)
            if cache["J"] is None:
                if cache["valid"]:
                    J = -np.asarray(_pw_fit_jac(jnp.asarray(x), iota, self._phi_grid,
                                                self._theta_grid, **statics)
                                    ).reshape(-1, np.size(x))/rms_modB
                    cache["J"] = J if np.all(np.isfinite(J)) else np.zeros((coil_modB.size, x.size))
                else:
                    cache["J"] = np.zeros((coil_modB.size, np.size(x)))
            return cache["J"]

        def fun(x):
            r = residual(x)
            return 0.5*float(r @ r)

        def grad(x):
            r = residual(x)
            return jac(x).T @ r

        kwargs.setdefault("maxiter", 300)
        kwargs.setdefault("ftol", 1e-12)
        res = minimize(fun, x0, jac=grad, method="SLSQP", options=kwargs)
        if not np.isfinite(best["objective"]):
            raise RuntimeError("Piecewise target fit produced no finite iterate")
        fitted = best["x"]
        self._zeta_c = float(np.mod(fitted[5], 2*np.pi/self._nfp))
        self._theta_c = float(np.mod(fitted[6], 2*np.pi))
        self.set_dofs(fitted[:5])
        self.need_to_update = True
        self.set_recompute_flag()
        res.fit_x = fitted.copy()
        res.initial_centers = initial_centers
        res.fitted_centers = np.array([self._zeta_c, self._theta_c])
        res.x = fitted[:5].copy()
        res.fun = best["objective"]
        self.fit_res = res
        proc0_print(f"fit_to: nfev={tracker['nfev']}, objective={best['objective']:.6e} "
                    f"success={res.success}")
        return res
