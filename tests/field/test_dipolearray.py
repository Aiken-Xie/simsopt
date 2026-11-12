import json
import unittest

import numpy as np

from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON
from simsopt.field import BiotSavart, Coil, Current, DipoleArrayField, coils_via_symmetries
from simsopt.geo import CurveXYZFourier


def make_coils(nfp=2, stellsym=True, nq=40):
    """Symmetric coil set: 2 free base currents, 1 fixed base current (ScaledCurrent chains for the images)."""
    rng = np.random.default_rng(3)
    base_curves = []
    for k in range(3):
        c = CurveXYZFourier(nq, 2)
        d = np.zeros_like(c.get_dofs())
        d[:] = 0.05 * rng.standard_normal(d.shape)
        dm = c.dofs_matrix
        dm[0][0] = 1.2 + 0.3 * k  # x: cos(0) ... overall offset
        dm[1][1] = 0.4
        dm[2][2] = 0.4
        c.set_dofs(np.concatenate(dm) + d)
        c.fix_all()  # the array field has fixed geometry
        base_curves.append(c)
    currents = [Current(1e5 * (k + 1)) for k in range(3)]
    currents[2].fix_all()
    currents[0].local_lower_bounds = np.array([-5e5])
    currents[0].local_upper_bounds = np.array([5e5])
    return coils_via_symmetries(base_curves, currents, nfp, stellsym)


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


class TestDipoleArrayField(unittest.TestCase):

    def setUp(self):
        self.coils = make_coils()
        self.bs = BiotSavart(self.coils)
        self.field = DipoleArrayField.from_coils(self.coils)
        rng = np.random.default_rng(5)
        self.pts = np.ascontiguousarray(rng.standard_normal((25, 3)) * 0.4 + np.array([1.5, 0.2, 0.1]))
        self.rng = rng

    def test_dofs_match_biotsavart(self):
        self.assertTrue(np.array_equal(self.field.x, self.bs.x))
        self.assertTrue(np.array_equal(self.field.lower_bounds, self.bs.lower_bounds))
        self.assertTrue(np.array_equal(self.field.upper_bounds, self.bs.upper_bounds))
        self.assertEqual(len(self.field.coils), len(self.coils))
        for c, cf in zip(self.coils, self.field.coils):
            self.assertAlmostEqual(c.current.get_value(), cf.current.get_value(), places=8)

    def test_field_and_derivatives(self):
        self.bs.set_points(self.pts)
        self.field.set_points(self.pts)
        self.assertLess(rel(self.field.B(), self.bs.B()), 1e-13)
        self.assertLess(rel(self.field.dB_by_dX(), self.bs.dB_by_dX()), 1e-13)
        self.assertLess(rel(self.field.d2B_by_dXdX(), self.bs.d2B_by_dXdX()), 1e-13)

    def test_field_follows_dofs_and_points(self):
        x = self.bs.x * (1 + 0.2 * self.rng.standard_normal(self.bs.x.size))
        self.bs.set_points(self.pts)
        self.field.set_points(self.pts)
        B0 = self.field.B().copy()
        self.bs.x = x
        self.field.x = x
        self.assertGreater(rel(self.field.B(), B0), 1e-3)
        self.assertLess(rel(self.field.B(), self.bs.B()), 1e-13)
        pts2 = np.ascontiguousarray(self.pts[::-1] + 0.01)
        self.bs.set_points(pts2)
        self.field.set_points(pts2)
        self.assertLess(rel(self.field.dB_by_dX(), self.bs.dB_by_dX()), 1e-13)
        self.assertLess(rel(self.field.B(), self.bs.B()), 1e-13)

    def test_vjps_and_finite_differences(self):
        self.bs.set_points(self.pts)
        self.field.set_points(self.pts)
        v = self.rng.standard_normal(self.pts.shape)
        vg = self.rng.standard_normal((len(self.pts), 3, 3))
        go, gn = self.bs.B_vjp(v)(self.bs), self.field.B_vjp(v)(self.field)
        self.assertEqual(go.shape, gn.shape)
        self.assertLess(rel(gn, go), 1e-12)
        (o1, o2), (n1, n2) = self.bs.B_and_dB_vjp(v, vg), self.field.B_and_dB_vjp(v, vg)
        self.assertLess(rel(n1(self.field), o1(self.bs)), 1e-12)
        self.assertLess(rel(n2(self.field), o2(self.bs)), 1e-12)
        # Derivative objects can be summed like the old ones
        self.assertLess(rel((n1 + n2)(self.field), (o1 + o2)(self.bs)), 1e-12)
        # finite differences (the field is linear in the currents)
        x0 = np.array(self.field.x)
        for k in range(x0.size):
            h = 1.0
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            self.field.x = xp
            self.field.set_points(self.pts)
            fp = np.sum(v * self.field.B())
            self.field.x = xm
            self.field.set_points(self.pts)
            fm = np.sum(v * self.field.B())
            self.assertAlmostEqual((fp - fm) / (2 * h), gn[k], delta=1e-7 * np.max(np.abs(gn)))
        self.field.x = x0

    def test_json_round_trip_and_shared_currents(self):
        clone = self.field.clone()
        self.assertIs(clone.currents, self.field.currents)
        s = json.dumps(SIMSON([self.field, clone]), cls=GSONEncoder)
        f2, c2 = json.loads(s, cls=GSONDecoder)
        self.assertIs(f2.currents, c2.currents)
        self.assertTrue(np.array_equal(f2.x, self.field.x))
        self.assertTrue(np.array_equal(f2.lower_bounds, self.field.lower_bounds))
        self.field.set_points(self.pts)
        f2.set_points(self.pts)
        self.assertLess(rel(f2.B(), self.field.B()), 1e-13)
        # a change of the shared currents reaches both views
        f2.x = f2.x * 1.1
        c2.set_points(self.pts)
        self.assertLess(rel(c2.B(), f2.B()), 1e-15)
        self.assertGreater(rel(c2.B(), self.field.B()), 1e-3)


if __name__ == "__main__":
    unittest.main()
