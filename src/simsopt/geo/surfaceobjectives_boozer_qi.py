import enum
import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from scipy import optimize
from simsopt import field, geo
from simsopt._core import derivative, optimizable
from simsopt.objectives.utilities import forward_backward


def compute_bounce_points_residuals_pure(
    B: jax.Array,
    iota: jax.Array,
    quadpoints_theta: jax.Array,
    quadpoints_phi: jax.Array,
    nlambda: int,
    nfp: int,
) -> jax.Array:
    """Computes a bounce points residual for an omnigeneous field.

    Args:
        B: the magnetic field vector field on a surface as a (nphi, ntheta, 3) array.
        iota: the iota value on the surface.
        quadpoints_theta: the poloidal quadrature points.
        quadpoints_phi: the tororidal quadrature points.
        nlambda: the number of magnetic field strength contours to consider.
        nfp: the surface number of field periods.
    """
    ntheta = len(quadpoints_theta)
    nphi = len(quadpoints_phi)

    # modb is (nphi, ntheta) -> (ntheta, nphi)
    modb = jnp.linalg.norm(B, axis=-1).T

    theta1d = 2 * jnp.pi * quadpoints_theta
    phi1d = 2 * jnp.pi * quadpoints_phi

    # theta2d is (ntheta, nphi)
    alpha1d = jnp.linspace(0, 2 * jnp.pi, ntheta, endpoint=False)
    theta2d = alpha1d[:, None] + iota * phi1d[None, :]

    minimum_of_modb = jnp.min(modb)
    maximum_of_modb = jnp.max(modb)
    one_over_lambda = jnp.linspace(minimum_of_modb, maximum_of_modb, nlambda + 2)[1:-1]

    # modb_on_field_line is (ntheta, nphi)
    modb_on_field_line = vmap(
        lambda k: jnp.interp(
            x=jnp.mod(theta2d[:, k], 2 * jnp.pi), xp=theta1d, fp=modb[:, k]
        )
    )(jnp.arange(nphi))
    modb_on_field_line = jnp.array(modb_on_field_line).T

    # sign_of_d_modb_d_l_at_crossings is (ntheta, nphi, nlambda)
    sign_of_modb_minus_one_over_lambda = jnp.sign(
        modb_on_field_line[:, :, None] - one_over_lambda[None, None, :]
    )
    # Let us define that when modb = 1/lambda, the sign is 1.
    sign_of_modb_minus_one_over_lambda = jnp.where(
        sign_of_modb_minus_one_over_lambda == 0, 1, sign_of_modb_minus_one_over_lambda
    )
    sign_of_d_modb_d_l_at_bounce_points = (
        jnp.diff(sign_of_modb_minus_one_over_lambda, axis=1) / 2
    )

    # When padding with `0`, we assume that at the boundary there is no
    # bounce points.
    sign_of_d_modb_d_l_at_bounce_points = jnp.pad(
        sign_of_d_modb_d_l_at_bounce_points,
        ((0, 0), (0, 1), (0, 0)),
        constant_values=0.0,
    )

    # Interpolate phi of modb on field lines to obtain more precise
    # bounce point locations.
    # phi_at_bounce_points is (ntheta, nphi, nlambda)
    d_modb_d_phi = jnp.diff(modb_on_field_line, axis=1) / jnp.diff(phi1d)[None, :]
    d_modb_d_phi = jnp.pad(d_modb_d_phi, ((0, 0), (0, 1)), mode="edge")
    phi_at_bounce_points = (
        phi1d[None, :, None]
        + (one_over_lambda[None, None, :] - modb_on_field_line[:, :, None])
        / jnp.where(d_modb_d_phi == 0, 1, d_modb_d_phi)[:, :, None]
    )

    # Filter crossing points until the first negative index and after
    # the last positive index
    first_negative_index = (sign_of_d_modb_d_l_at_bounce_points < 0.0).argmax(axis=1)
    first_negative_index = jnp.where(
        jnp.all(sign_of_d_modb_d_l_at_bounce_points >= 0.0, axis=1),
        jnp.inf,
        first_negative_index,
    )
    last_positive_index = (
        nphi
        - jnp.flip((sign_of_d_modb_d_l_at_bounce_points > 0.0), axis=1).argmax(axis=1)
        - 1
    )
    last_positive_index = jnp.where(
        jnp.all(sign_of_d_modb_d_l_at_bounce_points <= 0.0, axis=1),
        -jnp.inf,
        last_positive_index,
    )

    non_valid_leading_bounce_points = (
        jnp.arange(nphi)[None, :, None] < first_negative_index[:, None, :]
    )
    non_valid_trailing_bounce_points = (
        jnp.arange(nphi)[None, :, None] > last_positive_index[:, None, :] + 1
    )
    valid_sign_of_d_modb_d_l_at_bounce_points = jnp.where(
        non_valid_leading_bounce_points | non_valid_trailing_bounce_points,
        0.0,
        sign_of_d_modb_d_l_at_bounce_points,
    )
    normalization_factor = 2 * jnp.pi / nfp
    normalized_phi_at_bounce_poitns = phi_at_bounce_points / normalization_factor

    valid_normalized_phi_at_bounce_points = (
        valid_sign_of_d_modb_d_l_at_bounce_points * normalized_phi_at_bounce_poitns
    )

    number_of_wells = (
        jnp.sum(valid_sign_of_d_modb_d_l_at_bounce_points != 0.0, axis=1) / 2
    )
    number_of_wells_at_lambda = jnp.sum(number_of_wells, axis=0)

    average_bounce_points_delta_at_lambda = jnp.sum(
        valid_normalized_phi_at_bounce_points,
        axis=(0, 1),
    ) / jnp.where(number_of_wells_at_lambda == 0, 1.0, number_of_wells_at_lambda)
    bounce_points_residuals = (
        jnp.sum(
            valid_normalized_phi_at_bounce_points,
            axis=1,
        )
        - average_bounce_points_delta_at_lambda[None, :]
    )

    return bounce_points_residuals.flatten() / jnp.sqrt(ntheta * nlambda)


def compute_bounce_points_residuals_total_pure(
    B: jax.Array,
    iota: jax.Array,
    quadpoints_theta: jax.Array,
    quadpoints_phi: jax.Array,
    nlambda: int,
    nfp: int,
) -> jax.Array:
    bounce_points_residuals = compute_bounce_points_residuals_pure(
        B=B,
        iota=iota,
        quadpoints_theta=quadpoints_theta,
        quadpoints_phi=quadpoints_phi,
        nlambda=nlambda,
        nfp=nfp,
    )
    return (bounce_points_residuals**2).sum()


def compute_residuals_minima_well_pure(
    B: jax.Array,
    iota: jax.Array,
    quadpoints_theta: jax.Array,
    quadpoints_phi: jax.Array,
    nlambda: int,
    nfp: int,
) -> jax.Array:
    """Computes a residual for the minima of the wells for an omnigeneous field. It is
    zero if all the minima are equal among each other.

    Only B is used, by the other arguments are included to keep the signature
       consistent with the one of compute_bounce_points_residuals_pure.

    Args:
        B: the magnetic field vector field on a surface as a (nphi, ntheta, 3) array.
        iota: the iota value on the surface.
        quadpoints_theta: the poloidal quadrature points.
        quadpoints_phi: the tororidal quadrature points.
        nlambda: the number of magnetic field strength contours to consider.
        nfp: the surface number of field periods.
    """
    # B is (nphi, ntheta, 3)
    modb = jnp.linalg.norm(B, axis=-1)
    minima = jnp.min(modb, axis=0)
    mean_of_minima = jnp.mean(minima)
    b00 = np.mean(modb)
    return ((minima - mean_of_minima) / b00).flatten() / jnp.sqrt(modb.shape[-1])


def compute_residuals_minima_well_total_pure(
    B: jax.Array,
    iota: jax.Array,
    quadpoints_theta: jax.Array,
    quadpoints_phi: jax.Array,
    nlambda: int,
    nfp: int,
) -> jax.Array:
    minima_residuals = compute_residuals_minima_well_pure(
        B=B,
        iota=iota,
        quadpoints_theta=quadpoints_theta,
        quadpoints_phi=quadpoints_phi,
        nlambda=nlambda,
        nfp=nfp,
    )
    return (minima_residuals**2).sum()


def compute_residuals_maxima_well_pure(
    B: jax.Array,
    iota: jax.Array,
    quadpoints_theta: jax.Array,
    quadpoints_phi: jax.Array,
    nlambda: int,
    nfp: int,
) -> jax.Array:
    """Computes a residual for the maxima of the wells for an omnigeneous field. It is
    zero if all the maxima are equal among each other.

    Only B is used, by the other arguments are included to keep the signature
       consistent with the one of compute_bounce_points_residuals_pure.

    Args:
        B: the magnetic field vector field on a surface as a (nphi, ntheta, 3) array.
        iota: the iota value on the surface.
        quadpoints_theta: the poloidal quadrature points.
        quadpoints_phi: the tororidal quadrature points.
        nlambda: the number of magnetic field strength contours to consider.
        nfp: the surface number of field periods.
    """

    # B is (nphi, ntheta, 3)
    modb = jnp.linalg.norm(B, axis=-1)
    maxima = jnp.max(modb, axis=0)
    mean_of_maxima = jnp.mean(maxima)
    b00 = np.mean(modb)
    return ((maxima - mean_of_maxima) / b00).flatten() / jnp.sqrt(modb.shape[-1])


def compute_residuals_maxima_well_total_pure(
    B: jax.Array,
    iota: jax.Array,
    quadpoints_theta: jax.Array,
    quadpoints_phi: jax.Array,
    nlambda: int,
    nfp: int,
) -> jax.Array:
    maxima_residuals = compute_residuals_maxima_well_pure(
        B=B,
        iota=iota,
        quadpoints_theta=quadpoints_theta,
        quadpoints_phi=quadpoints_phi,
        nlambda=nlambda,
        nfp=nfp,
    )
    return (maxima_residuals**2).sum()


def compute_residuals_modB_max_from_zero_well_pure(
    B: jax.Array,
    iota: jax.Array,
    quadpoints_theta: jax.Array,
    quadpoints_phi: jax.Array,
    nlambda: int,
    nfp: int,
) -> jax.Array:
    """Computes a residual for the distance between the maxima of the wells and value at
    toroidal angle = 0 (in Boozer coordiantes) for an omnigeneous field. It is zero if
    all the maxima aligns with toroidal angle = 0 in boozer coordiantes.

    Only B is used, by the other arguments are included to keep the signature
       consistent with the one of compute_bounce_points_residuals_pure.

    Args:
        B: the magnetic field vector field on a surface as a (nphi, ntheta, 3) array.
        iota: the iota value on the surface.
        quadpoints_theta: the poloidal quadrature points.
        quadpoints_phi: the tororidal quadrature points.
        nlambda: the number of magnetic field strength contours to consider.
        nfp: the surface number of field periods.
    """

    # B is (nphi, ntheta, 3)
    modb = jnp.linalg.norm(B, axis=-1)

    modb_at_0 = modb[0, :]
    maxima = np.max(modb, axis=0)
    b00 = np.mean(modb)
    return ((modb_at_0 - maxima) / b00).flatten() / jnp.sqrt(modb.shape[-1])


def compute_residuals_modB_max_from_zero_well_total_pure(
    B: jax.Array,
    iota: jax.Array,
    quadpoints_theta: jax.Array,
    quadpoints_phi: jax.Array,
    nlambda: int,
    nfp: int,
) -> jax.Array:
    modB_max_residuals = compute_residuals_modB_max_from_zero_well_pure(
        B=B,
        iota=iota,
        quadpoints_theta=quadpoints_theta,
        quadpoints_phi=quadpoints_phi,
        nlambda=nlambda,
        nfp=nfp,
    )
    return (modB_max_residuals**2).sum()


def compute_qi_residuals_total_pure(
    B: jax.Array,
    iota: jax.Array,
    quadpoints_theta: jax.Array,
    quadpoints_phi: jax.Array,
    nlambda: int,
    nfp: int,
) -> jax.Array:
    """Computes a set of residuals that should be zero for a quasi-isodynamic field."""
    all_residuals = jnp.concatenate(
        [
            compute_bounce_points_residuals_pure(
                B=B,
                iota=iota,
                quadpoints_theta=quadpoints_theta,
                quadpoints_phi=quadpoints_phi,
                nlambda=nlambda,
                nfp=nfp,
            ),
            compute_residuals_maxima_well_pure(
                B=B,
                iota=iota,
                quadpoints_theta=quadpoints_theta,
                quadpoints_phi=quadpoints_phi,
                nlambda=nlambda,
                nfp=nfp,
            ),
            compute_residuals_minima_well_pure(
                B=B,
                iota=iota,
                quadpoints_theta=quadpoints_theta,
                quadpoints_phi=quadpoints_phi,
                nlambda=nlambda,
                nfp=nfp,
            ),
            compute_residuals_modB_max_from_zero_well_pure(
                B=B,
                iota=iota,
                quadpoints_theta=quadpoints_theta,
                quadpoints_phi=quadpoints_phi,
                nlambda=nlambda,
                nfp=nfp,
            ),
        ]
    )
    return jnp.sum(all_residuals**2)


_QI_DIAGNOSTIC = Callable[[jax.Array, jax.Array, jax.Array, jax.Array, int, int], jax.Array]


class QuasiIsodynamicityObjectiveFunctions(enum.Enum):
    bounce_points: _QI_DIAGNOSTIC = functools.partial(
        compute_bounce_points_residuals_total_pure
    )
    minima_well: _QI_DIAGNOSTIC = functools.partial(
        compute_residuals_minima_well_total_pure
    )
    maxima_well: _QI_DIAGNOSTIC = functools.partial(
        compute_residuals_maxima_well_total_pure
    )
    maxima_from_zero: _QI_DIAGNOSTIC = functools.partial(
        compute_residuals_modB_max_from_zero_well_total_pure
    )
    total: _QI_DIAGNOSTIC = functools.partial(compute_qi_residuals_total_pure)


class QuasiIsodynamicityResidual(optimizable.Optimizable):
    """This objective computes a residual for a quasi-isodynamic field.

    Args:
        boozer_surface: The boozer surface on which to compute the residual.
        biot_savart: The `field.BiotSavart` object from where to get the
            field information.
        objective_function_pure: A function that specifies which residual to compute.
        ntheta: The number of poloidal quadrature points where to compute
            the magnetic field. This translates also into the number of field
            lines where to compute the residual. Defaults to 128.
        nphi: The number of toroidal quadrature points. Defaults to 128.
        nlambda: The number of B field contours to consider in the bounce points
            residual. Defaults to 128.
    """

    _boozer_surface: geo.BoozerSurface
    _biot_savart: field.BiotSavart
    _auxiliary_surface: geo.SurfaceXYZTensorFourier
    _nlambda: int
    _J: float | None
    _dJ: np.ndarray | None

    def __init__(
        self,
        boozer_surface: geo.BoozerSurface,
        biot_savart: field.BiotSavart,
        objective_function_pure: _QI_DIAGNOSTIC,
        ntheta: int = 128,
        nphi: int = 101,
        nlambda: int = 134,
    ) -> None:
        assert (
            boozer_surface.res["type"] == "exact"
        ), "Only boozer surfaces that exactly solve the residual are supported"
        assert (
            type(boozer_surface.surface) is geo.SurfaceXYZTensorFourier
        ), "Only SurfaceXYZTensorFourier is supported"

        optimizable.Optimizable.__init__(self, depends_on=[boozer_surface])

        # Set toroidal quadpoints such that we get the maximum of the
        # magnetic field at index = 0.
        surface = boozer_surface.surface
        biot_savart.set_points(surface.gamma().reshape((-1, 3)))
        B = biot_savart.B()
        B = B.reshape((surface.quadpoints_phi.size, surface.quadpoints_theta.size, 3))
        modb = np.linalg.norm(B, axis=-1)
        if np.mean(np.argmax(modb, axis=0)) == 0:
            phi_start = 0
        else:
            phi_start = -1 / (2 * int(surface.nfp))
        phis = np.linspace(
            phi_start,
            phi_start + 1 / int(surface.nfp),
            nphi,
            endpoint=False,
        )
        thetas = np.linspace(0, 1.0, ntheta, endpoint=False)

        auxiliary_surface = geo.SurfaceXYZTensorFourier(
            mpol=surface.mpol,
            ntor=surface.ntor,
            stellsym=surface.stellsym,
            nfp=int(surface.nfp),
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
            dofs=surface.dofs,
        )

        self._boozer_surface = boozer_surface
        self._biot_savart = biot_savart
        self._auxiliary_surface = auxiliary_surface
        self._nlambda = nlambda
        self.recompute_bell()

        partial_J_fun = functools.partial(
            objective_function_pure,
            quadpoints_theta=jnp.array(self._auxiliary_surface.quadpoints_theta),  # type: ignore
            quadpoints_phi=jnp.array(self._auxiliary_surface.quadpoints_phi),  # type: ignore
            nlambda=self._nlambda,  # type: ignore
            nfp=int(self._auxiliary_surface.nfp),  # type: ignore
        )
        self._J_fun = jit(partial_J_fun)
        self._dJ_by_dB_fun = jit(grad(partial_J_fun, argnums=0))
        self._dJ_by_diota_fun = jit(grad(partial_J_fun, argnums=1))

    def recompute_bell(self, parent: optimizable.Optimizable | None = None) -> None:
        self._J = None
        self._dJ = None

    def J(self) -> float:
        if self._J is None:
            self._compute()
        if self._J is None:
            raise ValueError(f"{self.__class__.__name__}.J is None.")
        return self._J

    @derivative.derivative_dec
    def dJ(self) -> np.ndarray:
        if self._dJ is None:
            self._compute()
        if self._dJ is None:
            raise ValueError(f"{self.__class__.__name__}.dJ is None.")
        return self._dJ

    def _compute(self) -> None:
        if self._boozer_surface.need_to_run_code:
            res = self._boozer_surface.res
            # TODO: can we avoid to hard-code parameters here?
            res = self._boozer_surface.solve_residual_equation_exactly_newton(
                tol=1e-13, maxiter=20, iota=res["iota"], G=res["G"]
            )

        self._biot_savart.set_points(self._auxiliary_surface.gamma().reshape((-1, 3)))

        nphi = self._auxiliary_surface.quadpoints_phi.size
        ntheta = self._auxiliary_surface.quadpoints_theta.size

        B = self._biot_savart.B()
        B = B.reshape((nphi, ntheta, 3))

        self._J = float(self._J_fun(B=B, iota=self._boozer_surface.res["iota"]))

        dJ_by_diota = self._dJ_by_diota_fun(
            B,
            self._boozer_surface.res["iota"],
        )
        dJ_by_dB = self._dJ_by_dB_fun(
            B,
            self._boozer_surface.res["iota"],
        )

        dB_by_dgamma = self._biot_savart.dB_by_dX().reshape((nphi, ntheta, 3, 3))

        dJ_by_dB = dJ_by_dB.reshape((-1, 3))
        dJ_by_dcoils = self._biot_savart.B_vjp(dJ_by_dB)

        # TODO: implement with VJP (?)
        dgamma_by_dsurfacecoefficients = self._auxiliary_surface.dgamma_by_dcoeff()
        # (nphi, ntheta, 3 (B), 3 (gamma), nsurfacecoefficients))
        dJ_by_dB = dJ_by_dB.reshape((nphi, ntheta, 3))
        dJ_by_dsurfacecoefficients = np.einsum(
            "ijk,ijkl,ijlm->m",
            dJ_by_dB,
            dB_by_dgamma,
            dgamma_by_dsurfacecoefficients,
            optimize=True,
        )

        iota = self._boozer_surface.res["iota"]
        G = self._boozer_surface.res["G"]
        P, L, U = self._boozer_surface.res["PLU"]
        dconstraint_dcoils_vjp = (
            geo.surfaceobjectives.boozer_surface_dexactresidual_dcoils_dcurrents_vjp
        )

        # dJ_by_dG = 0
        dJ_ds = np.concatenate(
            (dJ_by_dsurfacecoefficients, np.array([dJ_by_diota, 0.0]))
        )
        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, self._boozer_surface, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil


def minimize_boozer_penalty_constraints_ls(
    boozer_surface,
    tol,
    maxiter,
    constraint_weight,
    iota: float,
    iota_relative_bound: float,
) -> dict[str, Any]:
    if not boozer_surface.need_to_run_code:
        return boozer_surface.res

    surface = boozer_surface.surface
    x = np.concatenate((surface.get_dofs(), [iota]))

    # Set bounds for iota to avoid solution with iota values that are too
    # far from the initial guess.
    bounds = [[-np.inf] * x.shape[0], [np.inf] * x.shape[0]]
    bounds[0][-1] = iota - np.abs(iota) * iota_relative_bound
    bounds[1][-1] = iota + np.abs(iota) * iota_relative_bound

    def fun(x: np.ndarray) -> np.ndarray:
        return boozer_surface.boozer_penalty_constraints(
            x,
            derivatives=0,
            constraint_weight=constraint_weight,
            scalarize=False,
            optimize_G=False,
        )

    def jac(x: np.ndarray) -> np.ndarray:
        return boozer_surface.boozer_penalty_constraints(
            x,
            derivatives=1,
            constraint_weight=constraint_weight,
            scalarize=False,
            optimize_G=False,
        )[1]

    res = optimize.least_squares(
        fun,
        x,
        jac=jac,  # type: ignore
        ftol=tol,
        xtol=tol,
        gtol=tol,
        x_scale="jac",
        max_nfev=maxiter,
        verbose=2,
        bounds=bounds,
    )
    resdict = {
        "info": res,
        "residual": res.fun,
        "gradient": res.grad,
        "jacobian": res.jac,
        "success": res.status > 0,
        "G": None,
    }
    surface.set_dofs(res.x[:-1])
    resdict["s"] = surface
    resdict["iota"] = res.x[-1]

    boozer_surface.res = resdict
    boozer_surface.need_to_run_code = False
    return resdict
