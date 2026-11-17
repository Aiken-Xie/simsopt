import json
import unittest

import numpy as np

from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON
from simsopt.field import BiotSavart, Coil, Current, DipoleArrayField, coils_via_symmetries
from simsopt.field import apply_symmetries_to_curves
from simsopt.geo import (CurveXYZFourier, SurfaceRZFourier, CurveSurfaceDistance, CurveCurveDistance, CurvePlanarOnSurface,
                         CurveSuperEllipse, ScaledCurveSuperEllipse, RotatedCurve)


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

    def test_frozen_curves_in_curve_surface_distance(self):
        s = SurfaceRZFourier(nfp=2, stellsym=True, mpol=1, ntor=0,
                             quadpoints_phi=np.linspace(0, 0.5, 8, endpoint=False), quadpoints_theta=np.linspace(0, 1, 8, endpoint=False))
        s.set_rc(0, 0, 1.0)
        s.set_rc(1, 0, 0.3)
        s.set_zs(1, 0, 0.3)
        old = CurveSurfaceDistance([c.curve for c in self.coils], s, 0.5)
        new = CurveSurfaceDistance(self.field.curves, s, 0.5)
        self.assertAlmostEqual(old.J(), new.J(), places=12)
        self.assertLess(np.max(np.abs(old.dJ(partials=True)(s) - new.dJ(partials=True)(s))), 1e-10)


def make_dipoles(nphi=6, ntheta=4, free=True, seed=2):
    """Small dipole array: ntheta rows of CurvePlanarOnSurface (own a, b, n) rotated to nphi toroidal positions,
    plus a flipped copy of each row; currents: one free dof per row (+- sign for the flipped copy) and a fixed
    planar TF-like frozen coil.  Returns the 1:1 Coil list."""
    rng = np.random.default_rng(seed)
    surf = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=0,
                            quadpoints_phi=(np.arange(nphi) + 0.5) / nphi,
                            quadpoints_theta=np.arange(2 * ntheta) / (2 * ntheta))
    surf.set_rc(0, 0, 1.0)
    surf.set_rc(1, 0, 0.3)
    surf.set_zs(1, 0, 0.3)
    surf.fix_all()
    rows, currents = [], []
    for it in range(ntheta):
        ref = CurveSuperEllipse(24, 0.05, 0.08, 2.0)
        shape = ScaledCurveSuperEllipse(ref, shared_dofs=(), a=0.05 + 0.01 * rng.random(), b=0.07 + 0.02 * rng.random(),
                                        n=2.0 + 0.5 * rng.random())
        shape.set_lower_bound('n', 1.5)
        shape.set_upper_bound('n', 4.0)
        if not free:
            shape.fix_all()
        rows.append(CurvePlanarOnSurface(shape, surf, 0, it, align="theta"))
        currents.append(Current(1e4 * (it + 1)))
    coils = []
    for curve, cur in zip(rows, currents):
        coils += [Coil(c, cur) for c in apply_symmetries_to_curves([curve], nphi, False)]
        from simsopt.field.coil import ScaledCurrent
        flipped = RotatedCurve(curve, 0.0, True)
        coils += [Coil(c, ScaledCurrent(cur, -1.0)) for c in apply_symmetries_to_curves([flipped], nphi, False)]
    tf = CurveXYZFourier(24, 1)
    tf.set('xc(1)', 1.5)
    tf.set('ys(1)', 1.5)
    tf.fix_all()
    tfc = Current(5e4)
    tfc.fix_all()
    coils.append(Coil(tf, tfc))
    return coils


class TestDipoleArrayFreeShape(unittest.TestCase):

    def setUp(self):
        self.coils = make_dipoles()
        self.bs = BiotSavart(self.coils)
        self.field = DipoleArrayField.from_coils(self.coils, free_shape=True)
        rng = np.random.default_rng(7)
        cen = np.array([c.curve.gamma().mean(axis=0) for c in self.coils])
        self.pts = np.ascontiguousarray(cen[rng.integers(0, len(cen), 40)] + 0.1 * rng.standard_normal((40, 3)))
        self.rng = rng

    def perturbed_x(self):
        x = self.bs.x.copy()
        return x * (1 + 0.08 * self.rng.standard_normal(x.size))

    def test_structure_and_dofs(self):
        self.assertEqual(len(self.field.base_curves), 4)
        self.assertEqual(self.field.x.size, self.bs.x.size)
        self.assertEqual(self.field.x.size, 4 + 4 * 3)  # 4 free currents + a,b,n of 4 base curves
        self.assertTrue(np.array_equal(self.field.x, self.bs.x))
        self.assertTrue(np.array_equal(self.field.lower_bounds, self.bs.lower_bounds))
        self.assertTrue(np.array_equal(self.field.upper_bounds, self.bs.upper_bounds))
        # the TF-like coil is frozen, the dipoles are images of the 4 base curves
        self.assertEqual(int(np.sum(self.field.image_base < 0)), 1)
        # graph: the field has few ancestors, not one per coil
        self.assertLess(len(self.field.ancestors), 4 * 3 + 5)
        for c, cf in zip(self.coils, self.field.curves):
            self.assertLess(np.max(np.abs(c.curve.gamma() - cf.gamma())), 1e-14)
            self.assertLess(np.max(np.abs(c.curve.gammadash() - cf.gammadash())), 1e-13)

    def test_field_derivatives_and_vjps_vs_biotsavart(self):
        self.bs.set_points(self.pts)
        self.field.set_points(self.pts)
        for trial in range(2):
            if trial:
                x = self.perturbed_x()
                self.bs.x = x
                self.field.x = x
            self.assertLess(rel(self.field.B(), self.bs.B()), 1e-13)
            self.assertLess(rel(self.field.dB_by_dX(), self.bs.dB_by_dX()), 1e-13)
            self.assertLess(rel(self.field.d2B_by_dXdX(), self.bs.d2B_by_dXdX()), 1e-13)
            v = self.rng.standard_normal(self.pts.shape)
            vg = self.rng.standard_normal((len(self.pts), 3, 3))
            go, gn = self.bs.B_vjp(v)(self.bs), self.field.B_vjp(v)(self.field)
            self.assertEqual(go.shape, gn.shape)
            self.assertLess(rel(gn, go), 1e-11)
            (o1, o2), (n1, n2) = self.bs.B_and_dB_vjp(v, vg), self.field.B_and_dB_vjp(v, vg)
            self.assertLess(rel(n1(self.field), o1(self.bs)), 1e-11)
            self.assertLess(rel(n2(self.field), o2(self.bs)), 1e-11)
            self.assertLess(rel((n1 + n2)(self.field), (o1 + o2)(self.bs)), 1e-11)

    def test_shape_finite_differences(self):
        self.field.set_points(self.pts)
        v = self.rng.standard_normal(self.pts.shape)
        x0 = np.array(self.field.x)
        g = self.field.B_vjp(v)(self.field)
        for k in range(4, x0.size):
            h = 1e-6
            vals = []
            for sgn in (+1, -1):
                x = x0.copy()
                x[k] += sgn * h
                self.field.x = x
                vals.append(np.sum(v * self.field.B()))
            fd = (vals[0] - vals[1]) / (2 * h)
            self.assertAlmostEqual(fd, g[k], delta=1e-6 * np.max(np.abs(g[4:])))
        self.field.x = x0

    def test_current_only_change_reuses_unit_fields_and_shape_change_recomputes_images(self):
        self.field.set_points(self.pts)
        self.field.B()
        calls = []
        orig = DipoleArrayField._unit_fields
        def spy(s, idx, pts, derivatives):
            calls.append(len(list(idx)))
            return orig(s, idx, pts, derivatives)
        DipoleArrayField._unit_fields = spy
        try:
            x = self.field.x.copy()
            x[:4] *= 1.1
            self.field.x = x
            self.field.B()
            self.assertEqual(calls, [])             # currents only: cached unit fields reused
            x = self.field.x.copy()
            x[4:7] *= 1.05                          # shape of base curve 0 only
            self.field.x = x
            self.field.B()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], len(self.field._base_members[0]))
        finally:
            DipoleArrayField._unit_fields = orig
        self.bs.x = self.field.x
        self.bs.set_points(self.pts)
        self.assertLess(rel(self.field.B(), self.bs.B()), 1e-13)

    def test_json_round_trip(self):
        s = json.dumps(SIMSON([self.field]), cls=GSONEncoder)
        (f2,) = json.loads(s, cls=GSONDecoder)
        self.assertTrue(np.array_equal(f2.x, self.field.x))
        self.assertTrue(np.array_equal(f2.lower_bounds, self.field.lower_bounds))
        self.assertEqual(len(f2.base_curves), 4)
        self.field.set_points(self.pts)
        f2.set_points(self.pts)
        self.assertLess(rel(f2.B(), self.field.B()), 1e-13)
        x = self.perturbed_x()
        f2.x = x
        self.field.x = x
        v = self.rng.standard_normal(self.pts.shape)
        self.assertLess(rel(f2.B(), self.field.B()), 1e-13)
        self.assertLess(rel(f2.B_vjp(v)(f2), self.field.B_vjp(v)(self.field)), 1e-12)

    def test_image_curves_in_curve_surface_distance(self):
        s = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=0,
                             quadpoints_phi=np.linspace(0, 1, 12, endpoint=False), quadpoints_theta=np.linspace(0, 1, 8, endpoint=False))
        s.set_rc(0, 0, 1.0)
        s.set_rc(1, 0, 0.25)
        s.set_zs(1, 0, 0.25)
        s.fix_all()
        dip = [i for i in range(len(self.coils) - 1)]
        old = CurveSurfaceDistance([self.coils[i].curve for i in dip], s, 0.2)
        new = CurveSurfaceDistance([self.field.curves[i] for i in dip], s, 0.2)
        self.assertAlmostEqual(old.J(), new.J(), places=12)
        self.assertGreater(old.J(), 0)
        self.assertEqual(new.x.size, self.field.x.size - 4)  # shape dofs only (currents are not curve dofs)
        # the gradient w.r.t. the base shape dofs equals the one through the real coil curves
        go = dict(zip(old.dof_names, old.dJ()))
        gn = dict(zip(new.dof_names, new.dJ()))
        self.assertEqual(set(go), set(gn))
        scale = max(abs(v) for v in go.values())
        self.assertGreater(scale, 0)
        self.assertLess(max(abs(go[k] - gn[k]) for k in go), 1e-9 * scale)
        # and it follows a shape change of the field's base curves
        x = self.perturbed_x()
        self.field.x = x
        self.bs.x = x
        self.assertAlmostEqual(old.J(), new.J(), places=12)

    def test_image_curves_in_curve_curve_distance(self):
        """Dipole-dipole overlap penalty on the symmetry images: value and shape gradient equal the real-curve ones."""
        dip = list(range(len(self.coils) - 1))
        for dmin in (0.2, 1e-4):  # many candidate pairs / none
            old = CurveCurveDistance([self.coils[i].curve for i in dip], dmin)
            new = CurveCurveDistance([self.field.curves[i] for i in dip], dmin)
            self.assertAlmostEqual(old.J(), new.J(), places=12)
            go = dict(zip(old.dof_names, old.dJ()))
            gn = dict(zip(new.dof_names, new.dJ()))
            self.assertEqual(set(go), set(gn))
            scale = max(max(abs(v) for v in go.values()), 1e-300)
            self.assertLess(max(abs(go[k] - gn[k]) for k in go), 1e-9 * scale)
            if dmin == 0.2:
                self.assertGreater(old.J(), 0)
        x = self.perturbed_x()
        self.field.x = x
        self.bs.x = x
        old = CurveCurveDistance([self.coils[i].curve for i in dip], 0.2)
        new = CurveCurveDistance([self.field.curves[i] for i in dip], 0.2)
        self.assertAlmostEqual(old.J(), new.J(), places=12)

    def _loop_vs_batched(self, fn):
        """Run ``fn()`` with the batched vjp path and with it disabled (per-curve loop); return both results."""
        import simsopt.field.dipolearray as da
        batched = fn()
        orig = da._batchable
        da._batchable = lambda bases: False
        try:
            looped = fn()
        finally:
            da._batchable = orig
        return batched, looped

    def test_batched_vjp_matches_per_curve_loop(self):
        import simsopt.field.dipolearray as da
        self.assertTrue(da._batchable(self.field.base_curves))
        self.field.set_points(self.pts)
        s = SurfaceRZFourier(nfp=1, stellsym=True, mpol=1, ntor=0,
                             quadpoints_phi=np.linspace(0, 1, 12, endpoint=False), quadpoints_theta=np.linspace(0, 1, 8, endpoint=False))
        s.set_rc(0, 0, 1.0)
        s.set_rc(1, 0, 0.25)
        s.set_zs(1, 0, 0.25)
        s.fix_all()
        dip = list(range(len(self.coils) - 1))
        for trial in range(2):
            if trial:
                self.field.x = self.perturbed_x()
            v = self.rng.standard_normal(self.pts.shape)
            vg = self.rng.standard_normal((len(self.pts), 3, 3))
            a, b = self._loop_vs_batched(lambda: self.field.B_vjp(v)(self.field))
            self.assertLess(rel(a, b), 1e-12)
            (a1, a2), (b1, b2) = self._loop_vs_batched(
                lambda: tuple(d(self.field) for d in self.field.B_and_dB_vjp(v, vg)))
            self.assertLess(rel(a1, b1), 1e-12)
            self.assertLess(rel(a2, b2), 1e-12)
            for pen in (lambda: CurveSurfaceDistance([self.field.curves[i] for i in dip], s, 0.2),
                        lambda: CurveCurveDistance([self.field.curves[i] for i in dip], 0.2)):
                a, b = self._loop_vs_batched(lambda: np.array(pen().dJ()))
                self.assertGreater(np.max(np.abs(b)), 0)
                self.assertLess(rel(a, b), 1e-12)

    def test_batched_vjp_fallback_with_partly_fixed_dofs(self):
        """One base curve with a fixed dof (a slot mismatch would show) and the unbatchable fallback are still correct."""
        import simsopt.field.dipolearray as da
        coils = make_dipoles()
        field = DipoleArrayField.from_coils(coils, free_shape=True)
        base = field.base_curves[0]
        base.curve.fix('b')  # 'b' of the first base curve
        bs = BiotSavart(coils)
        self.assertEqual(field.x.size, bs.x.size)
        field.set_points(self.pts)
        bs.set_points(self.pts)
        x = bs.x * (1 + 0.05 * self.rng.standard_normal(bs.x.size))
        field.x = x
        bs.x = x
        v = self.rng.standard_normal(self.pts.shape)
        a, b = self._loop_vs_batched(lambda: field.B_vjp(v)(field))
        self.assertLess(rel(a, b), 1e-12)
        self.assertLess(rel(a, bs.B_vjp(v)(bs)), 1e-11)
        # unbatchable: bases of mixed type fall back to the loop
        self.assertIsNone(da.batched_base_vjps(field.base_curves[:1], [np.zeros((24, 3))], [np.zeros((24, 3))]))


if __name__ == "__main__":
    unittest.main()
