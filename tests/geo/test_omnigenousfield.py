import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from simsopt.geo.omnigenousfield import OmnigenousField

OLD_COMMIT = "46e7ce9f"
PRE_OFFSET_COMMIT = "763a603f"  # before the zeta_c / theta_c offset was added

sys.path.insert(0, os.path.dirname(__file__))
from omni_container_helpers import coil_modB, load_container, taylor_table  # noqa: E402


def _load_old_class(commit=None):
    """Import the unmodified OmnigenousField from the pre-change commit via git show."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src = subprocess.run(
        ["git", "-C", repo, "show", f"{commit or OLD_COMMIT}:src/simsopt/geo/omnigenousfield.py"],
        capture_output=True, text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "old_omnigenousfield.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("old_omnigenousfield", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OmnigenousField


def _qi_drivers_field():
    try:
        from qi_drivers.drivers import omnigenousfield as m
        return m
    except Exception:  # pragma: no cover
        return None


def _params(DN=3, sL=2, sN=3, seed=1):
    rng = np.random.default_rng(seed)
    return 0.15*rng.standard_normal(DN), 0.03*rng.standard_normal((sL, sN))


class TestHelicityConvention(unittest.TestCase):
    def test_per_period_bit_identical(self):
        Old = _load_old_class()
        Dp, sp = _params()
        for helicity, iota in [((1, 0), 0.7), ((0, 1), 0.6), ((1, 1), 0.45), ((0, 2), 0.6)]:
            kw = dict(Bmax=1.2, Bmin=0.9, nfp=3, iota=iota, Dparams=Dp, sparams=sp,
                      phi=16, theta=12, helicity=helicity)
            old = Old(**kw)
            new = OmnigenousField(**kw)
            self.assertEqual(new.helicity_convention, "per_period")
            self.assertTrue(np.array_equal(old.modB, new.modB), helicity)
            self.assertTrue(np.array_equal(old.dmodB_by_dparams(), new.dmodB_by_dparams()))
            self.assertTrue(np.array_equal(old.dmodB_by_diota(), new.dmodB_by_diota()))
            self.assertTrue(np.array_equal(old.dzeta_by_deta(), new.dzeta_by_deta()))
            self.assertTrue(np.array_equal(old.tilde_mat, new.tilde_mat))

    def test_full_torus_matches_qi_drivers(self):
        m = _qi_drivers_field()
        if m is None:
            self.skipTest("qi_drivers not importable")
        Dp, sp = _params()
        worst = 0.
        for nfp in (2, 4):
            phi = np.linspace(0, 2*np.pi/nfp, 14, endpoint=False)
            theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
            for helicity in [(1, 0), (0, 1), (1, 1), (1, nfp)]:
                iota = 0.6 if helicity[1] != nfp else 0.35
                new = OmnigenousField(1.2, 0.9, nfp, iota, Dp, sp, phi=phi, theta=theta,
                                      helicity=helicity, helicity_convention="full_torus")
                # qi_drivers' sfunc has sL+1 rows of x-modes; use the simsopt functions directly.
                ref = m.OmnigenousField(
                    1.2, 0.9, nfp, iota,
                    lambda x: new.Dfunc_pure(x, Dp), lambda x, y: new.sfunc_pure(x, y, sp),
                    phi=phi, theta=theta, helicity=helicity)
                self.assertEqual(new.Np, ref.Np)
                self.assertTrue(np.allclose(new.tilde_mat, ref.tilde_mat))
                self.assertAlmostEqual(new.tilde_iota, ref.tilde_iota, places=14)
                self.assertAlmostEqual(new.dtilde_iota_by_diota(),
                                       ref.d_tilde_iota_by_diota(), places=14)
                err = np.max(np.abs(new.modB - ref.modB))
                worst = max(worst, err)
                print(f"full_torus nfp={nfp} helicity={helicity}: max|modB diff| = {err:.3e}")
                self.assertLess(err, 1e-10)
        print(f"full_torus worst modB difference vs qi_drivers: {worst:.3e}")

    def test_full_torus_validation(self):
        kw = dict(Bmax=1.2, Bmin=0.9, iota=0.6, phi=8, theta=8, helicity_convention="full_torus")
        with self.assertRaises(ValueError):
            OmnigenousField(nfp=3, helicity=(1, 2), **kw)   # 2 does not divide 3
        with self.assertRaises(ValueError):
            OmnigenousField(nfp=4, helicity=(2, 2), **kw)   # not primitive
        with self.assertRaises(ValueError):
            OmnigenousField(nfp=4, helicity=(0, 0), **kw)
        with self.assertRaises(ValueError):
            OmnigenousField(nfp=4, helicity=(2, 0), **kw)   # not primitive, N=0
        OmnigenousField(nfp=4, helicity=(1, 4), **kw)
        with self.assertRaises(ValueError):
            OmnigenousField(1.2, 0.9, 3, 0.6, helicity_convention="bogus")
        # per_period is unchanged: (0, 2) fine with nfp = 3
        OmnigenousField(1.2, 0.9, 3, 0.6, helicity=(0, 2))

    def test_full_torus_derivatives_fd(self):
        Dp, sp = _params()
        for nfp, helicity, iota in [(4, (1, 2), 0.6), (4, (1, 4), 0.35), (2, (1, 0), 0.7)]:
            f = OmnigenousField(1.2, 0.9, nfp, iota, Dp, sp, phi=10, theta=9,
                                helicity=helicity, helicity_convention="full_torus")
            d = f.dmodB_by_diota()
            h = 1e-6
            f.set_iota(iota + h); up = f.modB.copy()
            f.set_iota(iota - h); dn = f.modB.copy()
            f.set_iota(iota)
            err = np.max(np.abs((up - dn)/(2*h) - d))
            self.assertLess(err, 1e-6)
            x = f.get_dofs(); j = 3
            dp = f.dmodB_by_dparams()[..., j]
            xp = x.copy(); xp[j] += h; f.set_dofs(xp); up = f.modB.copy()
            xm = x.copy(); xm[j] -= h; f.set_dofs(xm); dn = f.modB.copy()
            self.assertLess(np.max(np.abs((up - dn)/(2*h) - dp)), 1e-6)

    def test_roundtrip_convention(self):
        from simsopt import load, save
        Dp, sp = _params()
        for conv, helicity, nfp in [("full_torus", (1, 2), 4), ("per_period", (0, 2), 3)]:
            f = OmnigenousField(1.2, 0.9, nfp, 0.6, Dp, sp, phi=8, theta=8,
                                helicity=helicity, helicity_convention=conv)
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "f.json")
                save(f, path)
                g = load(path)
            self.assertEqual(g.helicity_convention, conv)
            self.assertEqual(g.helicity, helicity)
            self.assertTrue(np.array_equal(f.modB, g.modB))

    def test_old_file_without_key_loads_per_period(self):
        import json
        from simsopt import load, save
        f = OmnigenousField(1.2, 0.9, 3, 0.6, phi=8, theta=8, helicity=(0, 2))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.json")
            save(f, path)
            data = json.load(open(path))
            n = 0
            for obj in data["simsopt_objs"].values():
                if obj.get("@class") == "OmnigenousField":
                    obj.pop("helicity_convention"); n += 1
            self.assertEqual(n, 1)
            json.dump(data, open(path, "w"))
            g = load(path)
        self.assertEqual(g.helicity_convention, "per_period")
        self.assertTrue(np.array_equal(f.modB, g.modB))


class TestFitToWellMargin(unittest.TestCase):
    def _flat_ish_field(self):
        # B varies only weakly and not along the (1, 0) helicity angle, so the fit wants a flat well
        phi = np.linspace(0, 2*np.pi/2, 12, endpoint=False)
        theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
        P, T = np.meshgrid(phi, theta, indexing="ij")
        return phi, theta, 1.0 + 0.05*np.cos(T) + 0.01*np.sin(2*T + P)

    def test_fit_keeps_well_margin(self):
        import contextlib, io
        phi, theta, B = self._flat_ish_field()
        f = OmnigenousField(B.max(), B.min(), 2, 0.4, phi=phi, theta=theta, helicity=(1, 0))
        with contextlib.redirect_stdout(io.StringIO()):
            f.fit_to(B, slope_lb=0.1)
        self.assertGreaterEqual(f.Bmax - f.Bmin, 1e-3*np.mean(B)*(1 - 1e-9))

    def test_fit_starts_non_flat_from_flat_input(self):
        import contextlib, io
        phi, theta, _ = self._flat_ish_field()
        B = np.ones((phi.size, theta.size))
        f = OmnigenousField(1.0 + 1e-3, 1.0, 2, 0.4, phi=phi, theta=theta, helicity=(1, 0))
        with contextlib.redirect_stdout(io.StringIO()):
            f.fit_to(B, slope_lb=0.1)
        self.assertGreater(f.Bmax - f.Bmin, 0.0)

    def test_load_clamps_old_flat_well(self):
        import json
        import warnings
        from simsopt import load, save
        phi, theta, B = self._flat_ish_field()
        f = OmnigenousField(1.2, 1.0, 2, 0.4, phi=phi, theta=theta, helicity=(1, 0))
        for gap in (-6e-13, 9e-13):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "t.json")
                save(f, path)
                data = json.load(open(path))
                for obj in data["simsopt_objs"].values():
                    if obj.get("@class") == "DOFs":
                        obj["x"]["data"][0] = obj["x"]["data"][1] + gap
                    if obj.get("@class") == "OmnigenousField":
                        obj["Bmax"] = obj["Bmin"] + gap
                json.dump(data, open(path, "w"))
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    g = load(path)
                self.assertTrue(any("clamping" in str(x.message) for x in w))
                self.assertGreater(g.Bmax, g.Bmin)
                self.assertTrue(np.all(np.isfinite(g.modB)))

    def test_load_rejects_genuinely_inverted_well(self):
        import json
        from simsopt import load, save
        phi, theta, _ = self._flat_ish_field()
        f = OmnigenousField(1.2, 1.0, 2, 0.4, phi=phi, theta=theta, helicity=(1, 0))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            save(f, path)
            data = json.load(open(path))
            for obj in data["simsopt_objs"].values():
                if obj.get("@class") == "DOFs":
                    obj["x"]["data"][0] = 0.5
            json.dump(data, open(path, "w"))
            with self.assertRaises(ValueError):
                load(path)


class TestPhaseOffset(unittest.TestCase):
    def test_zero_offset_bit_identical_to_pre_offset_class(self):
        Old = _load_old_class(PRE_OFFSET_COMMIT)
        Dp, sp = _params()
        for conv, nfp in [("per_period", 4), ("full_torus", 4)]:
            for helicity, iota in [((1, 0), 0.7), ((0, 1), 0.6), ((1, 1), 0.45)]:
                kw = dict(Bmax=1.2, Bmin=0.9, nfp=nfp, iota=iota, Dparams=Dp, sparams=sp,
                          phi=16, theta=12, helicity=helicity, helicity_convention=conv)
                old = Old(**kw)
                for new in (OmnigenousField(**kw), OmnigenousField(zeta_c=0.0, theta_c=0.0, **kw)):
                    self.assertTrue(np.array_equal(old.modB, new.modB), (conv, helicity))
                    self.assertTrue(np.array_equal(old.dmodB_by_dparams(), new.dmodB_by_dparams()))
                    self.assertTrue(np.array_equal(old.dmodB_by_diota(), new.dmodB_by_diota()))
                    self.assertTrue(np.array_equal(old.dzeta_by_deta(), new.dzeta_by_deta()))
                    self.assertTrue(np.array_equal(old.dslope_by_dparams(), new.dslope_by_dparams()))

    def test_offset_is_a_shift_of_the_angles(self):
        Dp, sp = _params()
        # grids are multiples of the shift so the shifted field is a roll of the unshifted one
        nphi, ntheta, nfp = 12, 12, 3
        for helicity, zc, tc in [((1, 0), 0.0, np.pi), ((0, 1), 2*np.pi/nfp/3, 0.0),
                                 ((1, 1), 2*np.pi/nfp/3, np.pi/2)]:
            kw = dict(Bmax=1.2, Bmin=0.9, nfp=nfp, iota=0.45, Dparams=Dp, sparams=sp,
                      phi=nphi, theta=ntheta, helicity=helicity)
            base = OmnigenousField(**kw)
            shifted = OmnigenousField(zeta_c=zc, theta_c=tc, **kw)
            kphi = int(round(zc/(2*np.pi/nfp/nphi))); kth = int(round(tc/(2*np.pi/ntheta)))
            self.assertLess(np.max(np.abs(np.roll(base.modB, (kphi, kth), (0, 1)) - shifted.modB)), 1e-12)
            # derivatives follow the same shift
            self.assertLess(np.max(np.abs(np.roll(base.dmodB_by_dparams(), (kphi, kth), (0, 1))
                                          - shifted.dmodB_by_dparams())), 1e-9)
            self.assertLess(np.max(np.abs(np.roll(base.dmodB_by_diota(), (kphi, kth), (0, 1))
                                          - shifted.dmodB_by_diota())), 1e-9)

    def test_offset_derivatives_fd(self):
        Dp, sp = _params()
        for helicity in [(1, 0), (0, 1), (1, 1)]:
            f = OmnigenousField(1.2, 0.9, 3, 0.5, Dp, sp, phi=10, theta=9, helicity=helicity,
                                zeta_c=0.37, theta_c=2.9)
            h = 1e-6
            f.set_iota(0.5 + h); up = f.modB.copy()
            f.set_iota(0.5 - h); dn = f.modB.copy()
            f.set_iota(0.5)
            self.assertLess(np.max(np.abs((up - dn)/(2*h) - f.dmodB_by_diota())), 1e-6)
            x = f.get_dofs(); j = 4
            xp = x.copy(); xp[j] += h; f.set_dofs(xp); up = f.modB.copy()
            xm = x.copy(); xm[j] -= h; f.set_dofs(xm); dn = f.modB.copy()
            f.set_dofs(x)
            self.assertLess(np.max(np.abs((up - dn)/(2*h) - f.dmodB_by_dparams()[..., j])), 1e-6)

    def test_slopes_unchanged_by_pure_offset(self):
        Dp, sp = _params()
        kw = dict(Bmax=1.2, Bmin=0.9, nfp=2, iota=0.4, Dparams=Dp, sparams=sp, phi=12, theta=24,
                  helicity=(1, 0))
        a = OmnigenousField(**kw)
        b = OmnigenousField(theta_c=np.pi, **kw)  # pi is a multiple of the theta grid step
        self.assertTrue(np.allclose(np.sort(a.dzeta_by_deta()), np.sort(b.dzeta_by_deta()), atol=1e-12))
        self.assertTrue(np.allclose(np.sort(a.dslope_by_dparams(), axis=0),
                                    np.sort(b.dslope_by_dparams(), axis=0), atol=1e-12))

    def test_save_load_roundtrip_and_old_file(self):
        import json
        from simsopt import load, save
        Dp, sp = _params()
        f = OmnigenousField(1.2, 0.9, 3, 0.6, Dp, sp, phi=8, theta=8, helicity=(1, 1),
                            zeta_c=0.4, theta_c=3.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.json")
            save(f, path)
            g = load(path)
            self.assertEqual((g.zeta_c, g.theta_c), (0.4, 3.0))
            self.assertTrue(np.array_equal(f.modB, g.modB))
            data = json.load(open(path))
            for obj in data["simsopt_objs"].values():
                if obj.get("@class") == "OmnigenousField":
                    obj.pop("zeta_c"); obj.pop("theta_c")
            json.dump(data, open(path, "w"))
            h = load(path)
        self.assertEqual((h.zeta_c, h.theta_c), (0.0, 0.0))

    def _qa_field(self, tc, nphi=12, ntheta=24, nfp=2):
        phi = np.linspace(0, 2*np.pi/nfp, nphi, endpoint=False)
        theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        P, T = np.meshgrid(phi, theta, indexing="ij")
        return nfp, phi, theta, 1.0 + 0.06*(1 + np.cos(T - tc))/2 + 0.004*np.cos(2*(T - tc))

    def test_fit_qa_max_at_pi(self):
        import contextlib, io
        nfp, phi, theta, B = self._qa_field(np.pi)
        self.assertAlmostEqual(theta[np.argmax(B[0])], np.pi)
        f = OmnigenousField(B.max(), B.min(), nfp, 0.4, phi=phi, theta=theta, helicity=(1, 0))
        with contextlib.redirect_stdout(io.StringIO()):
            f.fit_to(B)
        self.assertAlmostEqual(f.theta_c, np.pi)
        self.assertEqual(f.zeta_c, 0.0)
        self.assertGreater(f.Bmax - f.Bmin, 0.5*(B.max() - B.min()))
        self.assertAlmostEqual(theta[np.argmax(f.modB[0])], np.pi)
        res = np.linalg.norm(B - f.modB)/np.linalg.norm(B)
        print(f"synthetic QA max at pi: residual {res:.3e}, Bmax-Bmin {f.Bmax-f.Bmin:.4e}")
        self.assertLess(res, 5e-3)

        # centre options
        g = OmnigenousField(B.max(), B.min(), nfp, 0.4, phi=phi, theta=theta, helicity=(1, 0))
        with contextlib.redirect_stdout(io.StringIO()):
            g.fit_to(B, center="none")
        self.assertEqual((g.zeta_c, g.theta_c), (0.0, 0.0))
        with contextlib.redirect_stdout(io.StringIO()):
            g.fit_to(B, center=(0.0, np.pi))
        self.assertEqual(g.theta_c, np.pi)
        with self.assertRaises(ValueError):
            g.fit_to(B, center="bogus")

    def test_center_rule_per_helicity(self):
        phi = np.linspace(0, 2*np.pi/3, 12, endpoint=False)
        theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
        B = np.zeros((12, 12)); B[5, 7] = 1.0
        for helicity, expect in [((1, 0), (0.0, theta[7])), ((0, 1), (phi[5], 0.0)),
                                 ((1, 1), (phi[5], theta[7]))]:
            f = OmnigenousField(1.2, 0.9, 3, 0.5, phi=phi, theta=theta, helicity=helicity)
            self.assertEqual(f.center_from_modB(B), expect)


class TestPhaseOffsetOnContainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.container = load_container()

    def _make_target(self, b, **fitkw):
        import contextlib, io
        from simsopt.geo import OmnigenousResidual
        s = b.surface
        f = OmnigenousField(1.2, 1.0, s.nfp, b.res["iota"], Dparams=np.zeros(3),
                            sparams=np.zeros((2, 3)), phi=s.quadpoints_phi*2*np.pi,
                            theta=s.quadpoints_theta*2*np.pi, helicity=(1, 0))
        r = OmnigenousResidual(b, f)
        with contextlib.redirect_stdout(io.StringIO()):
            r.fit_target_field(**fitkw)
        return r

    def test_before_after_table(self):
        Old = _load_old_class(PRE_OFFSET_COMMIT)
        print("\nhelicity (1,0), per-surface fit residual and Bmax-Bmin (before = 763a603f)")
        for k, b in enumerate(self.container.boozersurfaces):
            s = b.surface
            B = coil_modB(b)
            rel = lambda t: np.linalg.norm(B - t)/np.linalg.norm(B)
            kw = dict(nfp=s.nfp, iota=b.res["iota"], Dparams=np.zeros(3), sparams=np.zeros((2, 3)),
                      phi=s.quadpoints_phi*2*np.pi, theta=s.quadpoints_theta*2*np.pi, helicity=(1, 0))
            import contextlib, io
            old = Old(B.max(), B.min(), **kw)
            new = OmnigenousField(B.max(), B.min(), **kw)
            with contextlib.redirect_stdout(io.StringIO()):
                old.fit_to(B); new.fit_to(B)
            print(f"  surface {k}: before {rel(old.modB):.4e} (Bmax-Bmin {old.Bmax-old.Bmin:.3e})  "
                  f"after {rel(new.modB):.4e} (Bmax-Bmin {new.Bmax-new.Bmin:.3e}, "
                  f"zeta_c {new.zeta_c:.3f}, theta_c {new.theta_c:.3f})")
            self.assertLess(rel(new.modB), 0.015)
            self.assertLess(rel(new.modB), rel(old.modB))
            self.assertGreater(new.Bmax - new.Bmin, 1e-3)

    def test_slope_penalty_with_offset(self):
        from simsopt.geo import OmnigenousSlopePenalty
        b = self.container.boozersurfaces[-1]
        r = self._make_target(b)
        self.assertNotEqual(r.target.theta_c, 0.0)
        pen = OmnigenousSlopePenalty(b, r.target, slope_lb=0.1)
        self.assertTrue(np.isfinite(pen.J()))
        pen2 = OmnigenousSlopePenalty(b, r.target, slope_lb=1e3)  # force active shortfall
        self.assertGreater(pen2.J(), 0.0)
        self.assertTrue(np.all(np.isfinite(pen2.dJ(partials=True)(self.container))))
        r0 = OmnigenousField(1.2, 1.0, b.surface.nfp, b.res["iota"], Dparams=r.target.Dparams,
                             sparams=r.target.sparams, phi=r.target.phi, theta=r.target.theta,
                             helicity=(1, 0))
        r0.set_dofs(r.target.get_dofs())
        self.assertTrue(np.allclose(np.sort(r0.dzeta_by_deta()), np.sort(r.target.dzeta_by_deta()),
                                    atol=1e-12))

    def test_taylor_with_theta_c(self):
        residuals = [self._make_target(b) for b in self.container.boozersurfaces]
        self.assertTrue(all(abs(r.target.theta_c - np.pi) < 1e-12 for r in residuals))
        J0, rows = taylor_table(self.container, residuals)
        print(f"\nTaylor, theta_c = pi, sum J = {J0:.10f}")
        for eps, fd, adj, err in rows:
            print(f"  eps {eps:.0e}  central diff {fd: .10e}  adjoint {adj: .10e}  rel err {err:.3e}")
        # A fitted non-flat target makes the gradient small relative to J's higher derivatives, so
        # the relative error is larger than for a flat target, but it still falls as eps^2.
        self.assertLess(rows[0][3], 5e-4)
        self.assertLess(rows[1][3], rows[0][3]/50)
        self.assertLess(rows[2][3], 1e-7)


if __name__ == "__main__":
    unittest.main()
