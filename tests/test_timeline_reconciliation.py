import unittest
from datetime import datetime, timezone

from rampflow.bag_reconciliation import BagReconciler, Bag, Scan, BagState
from rampflow.passenger_connections import (
    ConnectionManager,
    Connection,
    ConnectionState,
)

UTC = timezone.utc
T = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


class LateScanTimelineTests(unittest.TestCase):
    def _loaded_bag(self):
        r = BagReconciler(max_late_scan_minutes=30)
        r.register(Bag("B1", "P1", "F2", BagState.ACCEPTED, T(8), "checkin", sequence=0))
        r.ingest(Scan("s-screen", "B1", T(8, 5), "screen", "screen"))
        r.ingest(Scan("s-sort", "B1", T(8, 10), "sort", "sort"))
        r.ingest(Scan("s-load", "B1", T(8, 30), "aircraft", "load", container_id="C9"))
        self.assertEqual(r.bags["B1"].state, BagState.LOADED)
        return r

    def test_late_missing_scan_does_not_roll_back_loaded(self):
        r = self._loaded_bag()
        # Late-arriving scan reporting an event that happened BEFORE loading.
        updated = r.ingest(Scan("s-late-missing", "B1", T(8, 20), "makeup", "missing"))
        self.assertEqual(updated.state, BagState.LOADED)
        self.assertEqual(updated.container_id, "C9")
        self.assertEqual(updated.location, "aircraft")
        # The late record is still retained as a revision of the log.
        self.assertEqual([s.scan_id for s in r._history["B1"]][-1], "s-late-missing")

    def test_late_offload_scan_revises_state_from_ordered_history(self):
        r = self._loaded_bag()
        # A late offload that actually happened after the load genuinely changes state.
        updated = r.ingest(Scan("s-late-offload", "B1", T(8, 40), "ramp", "offload"))
        self.assertEqual(updated.state, BagState.OFFLOADED)
        self.assertEqual(updated.location, "ramp")

    def test_scan_outside_tolerance_still_rejected(self):
        r = self._loaded_bag()
        with self.assertRaises(RuntimeError):
            r.ingest(Scan("s-too-old", "B1", T(7), "sort", "missing"))

    def test_replay_rejects_invalid_revision(self):
        r = self._loaded_bag()
        # 'load' cannot follow LOADED when replayed in actual-time order.
        with self.assertRaises(RuntimeError):
            r.ingest(Scan("s-double-load", "B1", T(8, 45), "aircraft", "load"))
        # Neither side of the timeline may change when the revision is invalid.
        self.assertEqual(len(r._history["B1"]), 3)
        self.assertEqual(r.bags["B1"].state, BagState.LOADED)

    def test_idempotent_scan_rescan_unchanged(self):
        r = self._loaded_bag()
        first = r.ingest(Scan("s-late-missing", "B1", T(8, 20), "makeup", "missing"))
        second = r.ingest(Scan("s-late-missing", "B1", T(8, 20), "makeup", "missing"))
        self.assertEqual(first, second)
        self.assertEqual(len(r._history["B1"]), 4)


class ConnectionTimingTests(unittest.TestCase):
    def test_terminal_change_and_assistance_are_additive(self):
        mgr = ConnectionManager(terminal_change_min=20)
        c = Connection(
            passenger_id="P1",
            inbound_flight="F1",
            outbound_flight="F2",
            inbound_onblock=T(8),
            outbound_offblock=T(9, 15),
            minimum_connect_min=45,
            assistance_min=15,
            same_terminal=False,
        )
        mgr.register(c)
        # Required: 45 + 20 (terminal) + 15 (assistance) = 80; available = 75.
        self.assertEqual(mgr.effective_minimum(c), 80)
        self.assertEqual(mgr.margin_minutes(c), -5)
        evaluated = mgr.evaluate(c)
        self.assertEqual(evaluated.state, ConnectionState.AT_RISK)

    def test_same_terminal_with_assistance_uses_assistance_only(self):
        mgr = ConnectionManager(terminal_change_min=20)
        c = Connection(
            "P2", "F1", "F3", T(8), T(9, 10),
            minimum_connect_min=45, assistance_min=10, same_terminal=True,
        )
        mgr.register(c)
        self.assertEqual(mgr.effective_minimum(c), 55)
        self.assertEqual(mgr.margin_minutes(c), 15)

    def test_cross_terminal_without_assistance(self):
        mgr = ConnectionManager(terminal_change_min=20)
        c = Connection(
            "P3", "F1", "F4", T(8), T(9, 5),
            minimum_connect_min=45, assistance_min=0, same_terminal=False,
        )
        mgr.register(c)
        self.assertEqual(mgr.effective_minimum(c), 65)
        self.assertEqual(mgr.margin_minutes(c), 0)


class AlignedTimelinesScenarioTests(unittest.TestCase):
    """Both timelines re-evaluated in actual-occurrence order must agree."""

    def test_no_false_passenger_ok_bag_missing_conclusion(self):
        # Passenger side: 75 minutes available, 45 MCT + 20 terminal + 15 assistance = 80 needed.
        mgr = ConnectionManager(terminal_change_min=20)
        conn = Connection(
            passenger_id="P1", inbound_flight="F1", outbound_flight="F2",
            inbound_onblock=T(8), outbound_offblock=T(9, 15),
            minimum_connect_min=45, assistance_min=15, same_terminal=False,
        )
        mgr.register(conn)
        self.assertEqual(mgr.evaluate(conn).state, ConnectionState.AT_RISK)

        # Bag side: late scan that predates the load must not undo LOADED.
        rec = BagReconciler()
        rec.register(Bag("B1", "P1", "F2", BagState.ACCEPTED, T(8), "checkin"))
        rec.ingest(Scan("a", "B1", T(8, 5), "screen", "screen"))
        rec.ingest(Scan("b", "B1", T(8, 10), "sort", "sort"))
        rec.ingest(Scan("c", "B1", T(8, 30), "aircraft", "load"))
        rec.ingest(Scan("d", "B1", T(8, 20), "makeup", "missing"))
        self.assertEqual(rec.bags["B1"].state, BagState.LOADED)
        result = rec.reconcile_departure("F2", ["P1"])
        self.assertIn("B1", result["loaded"])
        self.assertNotIn("B1", result["missing_for_boarded"])


if __name__ == "__main__":
    unittest.main()
