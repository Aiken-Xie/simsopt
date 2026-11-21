import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from simsopt.geo.omnigenousfield import OmnigenousField

OLD_COMMIT = "46e7ce9f"


def _load_old_class():
    """Import the unmodified OmnigenousField from the pre-change commit via git show."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src = subprocess.run(
        ["git", "-C", repo, "show", f"{OLD_COMMIT}:src/simsopt/geo/omnigenousfield.py"],
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


if __name__ == "__main__":
    unittest.main()
