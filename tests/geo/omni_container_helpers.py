"""Helpers for tests on a saved 5-surface BoozerLS `DipoleBoozerSurfaces` container.

The container path comes from the environment variable OMNI_TEST_CONTAINER; the tests that use it
are skipped if the file or dipole_drivers is not available.
"""
import contextlib
import io
import os
import unittest

import numpy as np

DEFAULT_CONTAINER = (
    "/private/tmp/claude-501/-Users-paullab-Desktop-Coil-Opt/c7812821-ffa2-4fe1-b52c-915813a0da6a/"
    "scratchpad/wataru_merge/s3ols_m6.boozersurfaces.json")


def container_path():
    return os.environ.get("OMNI_TEST_CONTAINER", DEFAULT_CONTAINER)


def load_container(quiet=True):
    path = container_path()
    if not os.path.exists(path):
        raise unittest.SkipTest(f"test container {path} not found")
    try:
        from simsopt import load
        with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
            container = load(path)
            for b in container.boozersurfaces:
                b.run_code(b.res["iota"], b.res["G"])
    except ImportError as e:  # dipole_drivers missing
        raise unittest.SkipTest(f"cannot load the container: {e}")
    return container


def coil_modB(b):
    s = b.surface
    b.biotsavart.set_points(s.gamma().reshape(-1, 3))
    return np.linalg.norm(b.biotsavart.B().reshape(s.gamma().shape), axis=-1)


def taylor_table(container, residuals, n_dofs=6, seed=3, epss=(1e-3, 1e-4, 1e-5), quiet=True):
    """Central-difference check of sum(residual.J) against the adjoint gradient.

    The direction is random over `n_dofs` dipole-current dofs of the container; every surface is
    re-solved (from its saved state) at each point. Returns a list of rows
    (eps, central difference, adjoint directional derivative, relative error).
    """
    boozersurfaces = list(container.boozersurfaces)
    base = [(b.surface.x.copy(), dict(b.res)) for b in boozersurfaces]
    x0 = container.x.copy()
    names = container.dof_names
    idx = [i for i, n in enumerate(names) if n.endswith(":x0")]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(idx, n_dofs, replace=False)
    h = np.zeros(x0.size)
    h[chosen] = rng.uniform(-1, 1, n_dofs)*np.maximum(np.abs(x0[chosen]), 1.0)

    def J(x):
        for b, (sx, res) in zip(boozersurfaces, base):
            b.surface.x = sx
            b.res = dict(res)
        container.x = x
        with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
            for b in boozersurfaces:
                b.run_code(b.res["iota"], b.res["G"])
        return sum(r.J() for r in residuals)

    J0 = J(x0)
    dJ = None
    for r in residuals:
        d = r.dJ(partials=True)
        dJ = d if dJ is None else dJ + d
    slope = float(dJ(container) @ h)
    rows = []
    for eps in epss:
        fd = (J(x0 + eps*h) - J(x0 - eps*h))/(2*eps)
        rows.append((eps, fd, slope, abs(fd - slope)/abs(slope)))
    J(x0)
    return J0, rows
