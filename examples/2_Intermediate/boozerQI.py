import pathlib

import numpy as np
from simsopt.geo import surfaceobjectives_boozer_qi as utils
from scipy import optimize
from simsopt import field, geo, objectives

DATA_FOLDER = pathlib.Path("tests/test_files")

# Select the configuration to optimize for.
# As of now, Boozer surfaces can be fitted only for the preciseQA configuration.

# The Stellaris configuration with a modified set of coils.
# VMEC_INPUT_FILE = DATA_FOLDER / "input.stellaris"
# MAKEGRID_INPUT_FILE = DATA_FOLDER / "coils.stellaris"
# IOTA = 0.99
# N_FIELD_PERIODS = 4

# A 3 field periods test configuration.
# VMEC_INPUT_FILE = DATA_FOLDER / "input.DM8t2ef9jdwyU2By39zHZmu"
# MAKEGRID_INPUT_FILE = DATA_FOLDER / "coils.DXSu5ioFeeLPi3ZQujvRuVo"
# IOTA = 0.72
# N_FIELD_PERIODS = 3

# The PreciseQA configuration from Landreman and Paul.
VMEC_INPUT_FILE = DATA_FOLDER / "input.LandremanPaul2021_QA"
MAKEGRID_INPUT_FILE = DATA_FOLDER / "coils.LandremanPaul2021_QA"
IOTA = 0.42
N_FIELD_PERIODS = 2

coils = field.load_coils_from_makegrid_file(
    filename=MAKEGRID_INPUT_FILE,
    order=10,
)
unique_coils = coils[:6]

all_curves = [c.curve for c in coils]
base_curves = [c.curve for c in unique_coils]
base_currents = [c.current for c in unique_coils]

biot_savart = field.BiotSavart(coils)
current_sum = N_FIELD_PERIODS * sum(abs(c.get_value()) for c in base_currents)
G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 10 ** (-7) / (2 * np.pi))

mpol = 8
ntor = 8
stellsym = True

phis = np.linspace(0, 1 / N_FIELD_PERIODS, 2 * ntor + 1, endpoint=False)
thetas = np.linspace(0, 1, 2 * mpol + 1, endpoint=False)

rz_fourier_surface = geo.SurfaceRZFourier.from_vmec_input(
    filename=VMEC_INPUT_FILE,
    quadpoints_theta=thetas,
    quadpoints_phi=phis,
)

squared_flux = objectives.SquaredFlux(
    surface=rz_fourier_surface, field=biot_savart, definition="local"
)
print(f"Coils local squared flux on target boundary: {squared_flux.J():.3e}")

surface = geo.SurfaceXYZTensorFourier(
    mpol=mpol,
    ntor=ntor,
    stellsym=stellsym,
    nfp=N_FIELD_PERIODS,
    quadpoints_phi=phis,
    quadpoints_theta=thetas,
)
surface.least_squares_fit(rz_fourier_surface.gamma())

volume = geo.Volume(surface)
target_volume = volume.J()

boozer_surface = geo.BoozerSurface(
    biotsavart=biot_savart,
    surface=surface,
    label=volume,
    targetlabel=target_volume,
)

# FIXME: it does not converge on a valid surface
boozer_surface_result = utils.minimize_boozer_penalty_constraints_ls(
    boozer_surface=boozer_surface,
    tol=1e-13,
    maxiter=50,
    constraint_weight=1.0,
    iota=IOTA,
    iota_relative_bound=0.1,
)
boozer_surface.need_to_run_code = True

boozer_surface_residual = geo.boozer_surface_residual(
    surface,
    boozer_surface_result["iota"],
    boozer_surface_result["G"],
    biot_savart,
    derivatives=0,
)[0]
print(
    f"LS {boozer_surface_result['info'].success}: "
    f"iota={boozer_surface_result['iota']:.3f}, vol={surface.volume():.3f}, "
    f"||residual||={np.linalg.norm(boozer_surface_residual):.3e}"
)

# FIXME: it does not converge on a valid surface
boozer_surface_result = boozer_surface.solve_residual_equation_exactly_newton(
    tol=1e-13,
    maxiter=20,
    iota=boozer_surface_result["iota"],
    G=boozer_surface_result["G"],
)
boozer_surface_residual = geo.boozer_surface_residual(
    surface,
    boozer_surface_result["iota"],
    boozer_surface_result["G"],
    biot_savart,
    derivatives=0,
)[0]
print(
    f"NEWTON {boozer_surface_result['success']}: iter={boozer_surface_result['iter']}, "
    f"iota={boozer_surface_result['iota']:.3f}, vol={surface.volume():.3f}, "
    f"||residual||={np.linalg.norm(boozer_surface_residual):.3e}"
)

major_radius = geo.MajorRadius(boozer_surface=boozer_surface)
curve_lengths = [geo.CurveLength(c) for c in base_curves]
minima_along_well = utils.QuasiIsodynamicityResidual(
    boozer_surface=boozer_surface,
    biot_savart=biot_savart,
    objective_function_pure=utils.QuasiIsodynamicityObjectiveFunctions.minima_well.value,
)

J_major_radius = objectives.QuadraticPenalty(
    obj=major_radius, cons=float(major_radius.J()), f="identity"
)
J_iotas = objectives.QuadraticPenalty(
    obj=geo.Iotas(boozer_surface), cons=float(boozer_surface.res["iota"]), f="identity"
)
J_curve_lengths = objectives.QuadraticPenalty(
    obj=sum(curve_lengths), cons=float(sum(curve_lengths).J()), f="max"
)

JF = J_iotas + J_major_radius + J_curve_lengths + minima_along_well

# let's fix the coil current
base_currents[0].fix_all()


# Mimic simpsopt example on the objective function.
def fun(dofs: np.ndarray) -> tuple[float, np.ndarray]:
    sdofs_prev = boozer_surface.surface.x
    iota_prev = boozer_surface.res["iota"]
    G_prev = boozer_surface.res["G"]

    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()

    if not boozer_surface.res["success"]:
        # failed, so reset back to previous surface and return a large value
        # of the objective.  The purpose is to trigger the line search to reduce
        # the step size.
        J = 1e3
        boozer_surface.surface.x = sdofs_prev
        boozer_surface.res["iota"] = iota_prev
        boozer_surface.res["G"] = G_prev

    cl_string = ", ".join([f"{J.J():.1f}" for J in curve_lengths])
    outstr = (
        f"J={J:.1e}, iota={boozer_surface.res['iota']:.2e}, mr={major_radius.J():.2e}"
    )
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in curve_lengths):.1f}"
    outstr += f", minima_along_well={minima_along_well.J():.1e}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)

    scale_factor = 1e-4
    return scale_factor * J, scale_factor * grad  # type: ignore


x0 = JF.x
result = optimize.minimize(
    fun=fun, x0=x0, jac=True, method="BFGS", options={"maxiter": 5}, tol=1e-15
)
print(result)
