import unittest
from datetime import datetime, timezone

from rampflow.bag_reconciliation import Bag, BagReconciler, BagState, Scan
from rampflow.passenger_connections import (
    Connection,
    ConnectionManager,
    ConnectionState,
)
from rampflow.baggage_connections import BaggageConnectionReplay


UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


class LateScanTimelineTests(unittest.TestCase):
    def _bag(self):
        return Bag(
            tag="BAG1",
            passenger_id="P1",
            flight_id="F2",
            state=BagState.ACCEPTED,
            last_seen_at=D(8),
            location="check-in",
        )

    def test_late_scan_does_not_regress_loaded_state(self):
        # Operational order: screen -> sort -> load, then a late scan that
        # actually happened between sort and load is reported afterwards.
        rec = BagReconciler(max_late_scan_minutes=120)
        rec.register(self._bag())
        rec.ingest(Scan("s1", "BAG1", D(8, 5), "security", "screen"))
        rec.ingest(Scan("s2", "BAG1", D(8, 15), "sort", "sort"))
        loaded = rec.ingest(Scan("s3", "BAG1", D(8, 25), "F2-hold", "load"))
        self.assertEqual(loaded.state, BagState.LOADED)

        # The old code applied this late 'missing' on top of LOADED and
        # regressed the bag to MISSING. Realigned timeline is
        # screen -> sort -> missing -> load, so the final fact is still LOADED.
        late = rec.ingest(Scan("s0", "BAG1", D(8, 20), "makeup-belt", "missing"))
        self.assertEqual(late.state, BagState.LOADED)
        self.assertEqual(late.location, "F2-hold")
        self.assertEqual(late.revision, 1)
        # History is exposed in actual-occurrence order.
        self.assertEqual(
            [s.scan_id for s in rec.history("BAG1")],
            ["s1", "s2", "s0", "s3"],
        )

    def test_late_missing_scan_between_sorting_steps_revises_forward(self):
        rec = BagReconciler(max_late_scan_minutes=120)
        rec.register(self._bag())
        rec.ingest(Scan("s1", "BAG1", D(8, 5), "security", "screen"))
        rec.ingest(Scan("s3", "BAG1", D(8, 25), "F2-hold", "sort"))
        # A late 'missing' reported at 8:10: real timeline is
        # screen -> missing -> sort, so replay revises the bag back through
        # MISSING and forward to SORTED (rather than regressing the floor view).
        revised = rec.ingest(Scan("s2", "BAG1", D(8, 10), "sort", "missing"))
        self.assertEqual(revised.state, BagState.SORTED)
        self.assertEqual(revised.revision, 1)

    def test_scan_beyond_tolerance_is_rejected(self):
        rec = BagReconciler(max_late_scan_minutes=30)
        rec.register(self._bag())
        rec.ingest(Scan("s1", "BAG1", D(9), "security", "screen"))
        with self.assertRaises(RuntimeError):
            rec.ingest(Scan("s0", "BAG1", D(8, 5), "security", "screen"))


class ConnectionMinimumTests(unittest.TestCase):
    def test_terminal_change_and_assistance_are_additive(self):
        mgr = ConnectionManager(terminal_change_min=20)
        conn = Connection(
            passenger_id="P1",
            inbound_flight="F1",
            outbound_flight="F2",
            inbound_onblock=D(10),
            outbound_offblock=D(11),
            minimum_connect_min=40,
            assistance_min=15,
            same_terminal=False,
        )
        # 40 base + 20 terminal change + 15 wheelchair assistance = 75.
        # The buggy max() returned 40 + max(15, 20) = 60.
        self.assertEqual(mgr.effective_minimum(conn), 75)

        same_terminal = Connection(
            passenger_id="P2",
            inbound_flight="F1",
            outbound_flight="F3",
            inbound_onblock=D(10),
            outbound_offblock=D(11),
            minimum_connect_min=40,
            assistance_min=15,
            same_terminal=True,
        )
        self.assertEqual(mgr.effective_minimum(same_terminal), 55)

    def test_undercount_flips_false_feasible_to_at_risk(self):
        mgr = ConnectionManager(terminal_change_min=20)
        conn = Connection(
            passenger_id="P1",
            inbound_flight="F1",
            outbound_flight="F2",
            inbound_onblock=D(10),
            outbound_offblock=D(11, 5),
            minimum_connect_min=40,
            assistance_min=15,
            same_terminal=False,
        )
        mgr.register(conn)
        evaluated = mgr.evaluate(conn)
        # Available 65; required 75 -> margin -10 -> at risk, not planned.
        self.assertEqual(evaluated.state, ConnectionState.AT_RISK)


class JointTimelineReplayTests(unittest.TestCase):
    def _setup(self):
        # Cross-terminal wheelchair connection, 65 minutes available.
        mgr = ConnectionManager(terminal_change_min=20)
        conn = Connection(
            passenger_id="P1",
            inbound_flight="F1",
            outbound_flight="F2",
            inbound_onblock=D(10),
            outbound_offblock=D(11, 5),
            minimum_connect_min=40,
            assistance_min=15,
            same_terminal=False,
        )
        mgr.register(conn)
        rec = BagReconciler(max_late_scan_minutes=180)
        rec.register(
            Bag("BAG1", "P1", "F2", BagState.ACCEPTED, D(10), "transfer-in")
        )
        replay = BaggageConnectionReplay(
            rec, mgr, {("P1", "F2"): ("BAG1",)}
        )
        return mgr, rec, replay, conn

    def test_both_timelines_realign_together(self):
        mgr, rec, replay, conn = self._setup()

        # What the floor believed at 10:50: bag already loaded at 10:40.
        rec.ingest(Scan("b1", "BAG1", D(10, 10), "transfer-security", "screen"))
        rec.ingest(Scan("b2", "BAG1", D(10, 25), "transfer-sort", "sort"))
        rec.ingest(Scan("b3", "BAG1", D(10, 40), "F2-hold", "load"))
        self.assertEqual(rec.bags["BAG1"].state, BagState.LOADED)

        # Under the two bugs this looked feasible for the passenger
        # (undercounted minimum) and missing for the bag (a late scan regressed
        # it). Realigned replay with a late intermediate scan: bag stays loaded.
        verdict = replay.replay(
            "P1", "F2",
            late_scans=(Scan("b0", "BAG1", D(10, 20), "makeup", "missing"),),
        )
        self.assertEqual(verdict.bags[0].state, BagState.LOADED)
        self.assertTrue(verdict.bags[0].feasible)

        # Passenger actually needs 40+20+15=75 min but only has 65: the joint
        # verdict must be MISCONNECTED rather than the old false "pax ok / bag
        # missing" split.
        self.assertEqual(verdict.passenger_margin_min, 65 - 75)
        self.assertFalse(verdict.passenger_feasible)
        self.assertEqual(verdict.joint_state, ConnectionState.MISCONNECTED)
        self.assertTrue(verdict.revised)

    def test_late_flight_update_and_scan_apply_to_both_sides(self):
        mgr, rec, replay, conn = self._setup()
        rec.ingest(Scan("b1", "BAG1", D(10, 10), "transfer-security", "screen"))
        rec.ingest(Scan("b2", "BAG1", D(10, 25), "transfer-sort", "sort"))
        rec.ingest(Scan("b3", "BAG1", D(10, 40), "F2-hold", "load"))

        # Outbound delayed to 11:30 (90 min available) and a late scan for the
        # bag are reported together; both timelines must be revised before one
        # verdict is produced.
        verdict = replay.replay(
            "P1", "F2",
            late_scans=(Scan("b0", "BAG1", D(10, 20), "makeup", "missing"),),
            outbound_offblock=D(11, 30),
        )
        self.assertEqual(verdict.passenger_margin_min, 90 - 75)
        self.assertTrue(verdict.feasible)
        self.assertTrue(verdict.bags[0].feasible)
        self.assertTrue(verdict.revised)


if __name__ == "__main__":
    unittest.main()
