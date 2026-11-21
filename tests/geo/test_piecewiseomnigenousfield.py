import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np

from simsopt import load, save
from simsopt.geo import OmnigenousResidual, PiecewiseOmnigenousField

sys.path.insert(0, os.path.dirname(__file__))
from omni_container_helpers import coil_modB, load_container, taylor_table  # noqa: E402


def _qi():
    try:
        from qi_drivers.drivers import omnigenousfield as m
        return m
    except Exception:  # pragma: no cover
        return None


def _grid(nfp, nphi=14, ntheta=12):
    return (np.linspace(0, 2*np.pi/nfp, nphi, endpoint=False),
            np.linspace(0, 2*np.pi, ntheta, endpoint=False))


# (nfp, iota, pmin, raw dofs [raw_gap, raw_min, q_f, q_k, raw_p], zeta_c, theta_c)
CASES = [
    (2, 0.30, 1.0, [0.2, 0.5, -2.0, -1.5, 0.1], None, None),
    (2, 0.30, 1.0, [-0.3, 0.9, 0.7, 0.4, 0.8], 0.6, 2.1),
    (3, 0.62, 1.0, [0.0, 0.3, 1.5, -0.5, -0.4], 0.3, 5.0),
    (4, -0.45, 1.0, [0.4, 0.2, -0.8, 1.1, 0.3], 0.2, 1.0),
    (5, 0.8, 2.0, [0.1, 0.6, 0.2, 0.9, 1.2], 0.5, 3.0),
]


def _pair(case):
    m = _qi()
    nfp, iota, pmin, x, zc, tc = case
    phi, theta = _grid(nfp)
    new = PiecewiseOmnigenousField(1.3, 1.0, nfp, iota, phi=phi, theta=theta, pmin=pmin)
    ref = m.PiecewiseOmnigenousField(1.3, 1.0, nfp, iota, phi=phi, theta=theta, pmin=pmin)
    new.set_dofs(x)
    ref.set_dofs(np.array(x))
    if zc is not None:
        new._zeta_c, new._theta_c = zc, tc
        ref.zeta_c, ref.theta_c = zc, tc
    new.need_to_update = True
    ref.need_to_update = True
    return new, ref


class TestPiecewiseOmnigenousField(unittest.TestCase):
    def test_matches_qi_drivers(self):
        if _qi() is None:
            self.skipTest("qi_drivers not importable")
        for case in CASES:
            new, ref = _pair(case)
            e0 = np.max(np.abs(new.modB - ref.modB))
            e1 = np.max(np.abs(new.dmodB_by_dparams() - ref.dmodB_by_dparams()))
            e2 = np.max(np.abs(new.dmodB_by_diota() - ref.dmodB_by_diota()))
            print(f"case nfp={case[0]} iota={case[1]}: modB {e0:.2e} dparams {e1:.2e} diota {e2:.2e}")
            self.assertLess(e0, 1e-12)
            self.assertLess(e1, 1e-12)
            self.assertLess(e2, 1e-12)
            for name in ("Bmax", "Bmin", "p"):
                self.assertAlmostEqual(getattr(new, name), getattr(ref, name), places=14)
            gn, gr = new.piecewise_geometry(), ref.piecewise_geometry()
            for key in gr:
                self.assertAlmostEqual(gn[key], gr[key], places=12, msg=key)
            self.assertTrue(np.allclose(new.physical_dofs(), ref.physical_dofs(), atol=1e-13))

    def test_matches_qi_drivers_physical_p(self):
        """Explicit exponents p = 2, 5, 10 with k > 0, through the physical parameterization."""
        if _qi() is None:
            self.skipTest("qi_drivers not importable")
        for p in (2.0, 5.0, 10.0):
            for nfp, iota, f, k in ((2, 0.3, 0.3, 0.15), (3, -0.5, 0.2, 0.2)):
                new, ref = _pair((nfp, iota, 1.0, [0.2, 0.5, 0.0, 0.0, 0.1], 0.7, 1.9))
                phys = [1.4, 0.9, f, k, p]
                new.set_physical_dofs(phys)
                ref.set_physical_dofs(np.array(phys))
                self.assertGreater(new.piecewise_geometry()["k"], 0.0)
                self.assertAlmostEqual(new.p, p, places=12)
                e0 = np.max(np.abs(new.modB - ref.modB))
                e1 = np.max(np.abs(new.dmodB_by_dparams() - ref.dmodB_by_dparams()))
                e2 = np.max(np.abs(new.dmodB_by_diota() - ref.dmodB_by_diota()))
                print(f"p={p} nfp={nfp} iota={iota}: modB {e0:.2e} dparams {e1:.2e} diota {e2:.2e}")
                self.assertLess(max(e0, e1, e2), 1e-12)

    def test_iota_reference_lock(self):
        if _qi() is None:
            self.skipTest("qi_drivers not importable")
        new, ref = _pair(CASES[0])
        new.lock_iota_reference(0.31)
        ref.lock_iota_reference(0.31)
        new.set_iota(0.35)
        ref.set_iota(0.35)
        self.assertEqual(new.iota, 0.31)
        self.assertTrue(np.array_equal(new.dmodB_by_diota(), np.zeros_like(new.modB)))
        self.assertLess(np.max(np.abs(new.modB - ref.modB)), 1e-12)
        new.unlock_iota_reference()
        new.set_iota(0.35)
        self.assertEqual(new.iota, 0.35)
        with self.assertRaises(ValueError):
            new.set_iota(-0.1)

    def test_diota_matches_finite_difference(self):
        new, _ = _pair(CASES[1]) if _qi() else (None, None)
        if new is None:
            phi, theta = _grid(2)
            new = PiecewiseOmnigenousField(1.3, 1.0, 2, 0.3, phi=phi, theta=theta)
        d = new.dmodB_by_diota()
        h = 1e-6
        i0 = new.iota
        new.set_iota(i0 + h); up = new.modB.copy()
        new.set_iota(i0 - h); dn = new.modB.copy()
        self.assertLess(np.max(np.abs((up - dn)/(2*h) - d)), 1e-6)

    def test_physical_dofs_roundtrip(self):
        phi, theta = _grid(2)
        f = PiecewiseOmnigenousField(1.4, 0.9, 2, 0.3, phi=phi, theta=theta)
        phys = np.array([1.5, 0.8, 0.2, 0.1, 2.5])
        f.set_physical_dofs(phys)
        self.assertTrue(np.allclose(f.physical_dofs(), phys, atol=1e-12))
        self.assertAlmostEqual(f.Bmax, 1.5, places=12)
        self.assertAlmostEqual(f.Bmin, 0.8, places=12)

    def test_fit_recovers_known_target(self):
        phi, theta = _grid(2, 20, 20)
        truth = PiecewiseOmnigenousField(1.4, 0.9, 2, 0.3, phi=phi, theta=theta)
        truth.set_physical_dofs([1.4, 0.9, 0.3, 0.15, 3.0])
        truth._zeta_c, truth._theta_c = 0.9, 1.7
        truth.need_to_update = True
        fit = PiecewiseOmnigenousField(1.4, 0.9, 2, 0.3, phi=phi, theta=theta)
        res = fit.fit_to(truth.modB)
        rel = np.linalg.norm(fit.modB - truth.modB)/np.linalg.norm(truth.modB)
        print(f"fit of known pwO target: relative error {rel:.2e}, fun={res.fun:.2e}")
        self.assertLess(rel, 5e-2)

    def test_save_load(self):
        phi, theta = _grid(3)
        f = PiecewiseOmnigenousField(1.3, 1.0, 3, 0.4, phi=phi, theta=theta, pmin=1.5)
        f.set_dofs([0.1, 0.4, 0.3, -0.2, 0.5])
        f._zeta_c, f._theta_c = 0.4, 2.2
        f.lock_iota_reference(0.4)
        f.fix("Bmax")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pw.json")
            save(f, path)
            g = load(path)
        self.assertIsInstance(g, PiecewiseOmnigenousField)
        self.assertTrue(np.array_equal(f.get_dofs(), g.get_dofs()))
        self.assertEqual((g.zeta_c, g.theta_c, g.iota_ref, g.pmin, g.nfp),
                         (0.4, 2.2, 0.4, 1.5, 3))
        self.assertTrue(np.array_equal(f.modB, g.modB))
        self.assertTrue(np.array_equal(f.dmodB_by_dparams(), g.dmodB_by_dparams()))
        self.assertTrue(g.is_fixed("Bmax"))
        self.assertEqual(g.x.size, 4)


def _fitted_residuals(container, lock):
    residuals = []
    for b in container.boozersurfaces:
        s = b.surface
        m = coil_modB(b)
        target = PiecewiseOmnigenousField(
            m.max(), m.min(), s.nfp, b.res["iota"],
            phi=2*np.pi*s.quadpoints_phi, theta=2*np.pi*s.quadpoints_theta)
        residual = OmnigenousResidual(b, target)
        with contextlib.redirect_stdout(io.StringIO()):
            residual.fit_target_field()
        if lock:
            target.lock_iota_reference(b.res["iota"])
        target.local_fix_all()  # the target dofs are frozen; the Taylor direction is on the coils
        residuals.append(residual)
    return residuals


class TestPiecewiseOnContainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.container = load_container()

    def test_fitted_outer_surface_matches_qi_drivers(self):
        if _qi() is None:
            self.skipTest("qi_drivers not importable")
        from qi_drivers.drivers.targets import build_target_field, fit_target_field
        b = self.container.boozersurfaces[-1]
        s = b.surface
        m = coil_modB(b)
        with contextlib.redirect_stdout(io.StringIO()):
            ref = build_target_field(b, 1, 1, 1, model="piecewise", pmin=1.0, loss="norm")
            fit_target_field(ref, slope_lb=0.1)
        new = PiecewiseOmnigenousField(
            m.max(), m.min(), s.nfp, b.res["iota"],
            phi=2*np.pi*s.quadpoints_phi, theta=2*np.pi*s.quadpoints_theta)
        with contextlib.redirect_stdout(io.StringIO()):
            new.fit_to(m)
        rel = lambda t: np.linalg.norm(m - t)/np.linalg.norm(m)
        print(f"outer-surface fit residual: simsopt {rel(new.modB):.12e}, qi_drivers {rel(ref.modB):.12e}")
        print(f"independent fits: max|modB diff| {np.max(np.abs(new.modB - ref.modB)):.2e}, "
              f"centres {new.zeta_c:.12f},{new.theta_c:.12f} vs {ref.zeta_c:.12f},{ref.theta_c:.12f}")
        self.assertLess(abs(rel(new.modB) - rel(ref.modB)), 1e-9)
        self.assertLess(np.max(np.abs(new.modB - ref.modB)), 1e-9)

        # Same parameters, both evaluators: modB and both derivatives to 1e-12.
        twin = PiecewiseOmnigenousField(
            m.max(), m.min(), s.nfp, b.res["iota"],
            phi=2*np.pi*s.quadpoints_phi, theta=2*np.pi*s.quadpoints_theta)
        twin.set_dofs(ref.get_dofs())
        twin._zeta_c, twin._theta_c = ref.zeta_c, ref.theta_c
        twin.need_to_update = True
        e0 = np.max(np.abs(twin.modB - ref.modB))
        e1 = np.max(np.abs(twin.dmodB_by_dparams() - ref.dmodB_by_dparams()))
        e2 = np.max(np.abs(twin.dmodB_by_diota() - ref.dmodB_by_diota()))
        print(f"fitted parameters: modB {e0:.2e}, dparams {e1:.2e}, diota {e2:.2e}")
        self.assertLess(max(e0, e1, e2), 1e-12)

    def _taylor(self, lock):
        residuals = _fitted_residuals(self.container, lock)
        J0, rows = taylor_table(self.container, residuals)
        print(f"\nTaylor, iota reference {'locked' if lock else 'free'}, sum J = {J0:.10f}")
        for eps, fd, adj, err in rows:
            print(f"  eps {eps:.0e}  central diff {fd: .10e}  adjoint {adj: .10e}  rel err {err:.3e}")
        self.assertLess(rows[1][3], 1e-8)
        self.assertLess(rows[2][3], 1e-8)
        self.assertLess(rows[1][3], rows[0][3])

    def test_taylor_free_iota(self):
        self._taylor(False)

    def test_taylor_locked_iota(self):
        self._taylor(True)


if __name__ == "__main__":
    unittest.main()
