"""Joint replay of passenger and baggage transfer timelines.

A late-arriving fact (a delayed scan, a revised on-block/off-block time) forms
a *revision* of the operational picture, never an append on top of the current
projection. This module replays the passenger connection timeline and the
baggage scan timeline together and derives one verdict, so the two sides
cannot disagree into the false "passenger makes it but the bag does not"
conclusion: a connection is only feasible when the passenger margin **and**
every checked bag's handling margin are non-negative against the same
realigned timelines.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping

from .bag_reconciliation import BagReconciler, BagState, Scan
from .passenger_connections import Connection, ConnectionManager, ConnectionState
from .timeutil import utc


@dataclass(frozen=True)
class BagLineItem:
    tag: str
    state: BagState
    last_seen_at: datetime
    ready_at: datetime
    deadline: datetime | None
    feasible: bool
    revision: int


@dataclass(frozen=True)
class JointConnectionVerdict:
    passenger_id: str
    outbound_flight: str
    connection: Connection
    passenger_ready_at: datetime
    passenger_deadline: datetime
    passenger_margin_min: int
    passenger_feasible: bool
    bags: tuple[BagLineItem, ...]
    revised: bool
    joint_state: ConnectionState

    @property
    def baggage_feasible(self) -> bool:
        return bool(self.bags) and all(item.feasible for item in self.bags)

    @property
    def feasible(self) -> bool:
        return self.passenger_feasible and self.baggage_feasible


# States in which a bag is still moving through the transfer flow rather
# than being secured on the outbound aircraft.
_BAG_IN_FLOW_STATES = {
    BagState.SORTED,
    BagState.SCREENED,
    BagState.TRANSFER,
}


class BaggageConnectionReplay:
    """Re-aligns the passenger and baggage timelines as one projection."""

    def __init__(
        self,
        reconciler: BagReconciler,
        connections: ConnectionManager,
        bag_tags: Mapping[tuple[str, str], tuple[str, ...]],
        baggage_cutoff_min: int = 20,
    ):
        self.reconciler = reconciler
        self.connections = connections
        # (passenger_id, outbound_flight) -> checked bag tags
        self.bag_tags = {key: tuple(tags) for key, tags in bag_tags.items()}
        # Bags still in the flow must reach their final handling step by this
        # many minutes before off-block. A bag already LOADED is physically
        # aboard the outbound aircraft and has no further cut-off.
        self.baggage_cutoff_min = baggage_cutoff_min

    def _key(self, passenger_id: str, outbound_flight: str) -> str:
        return f"{passenger_id}:{outbound_flight}"

    def evaluate(self, passenger_id: str, outbound_flight: str) -> JointConnectionVerdict:
        stored = self.connections.connections[self._key(passenger_id, outbound_flight)]
        # evaluate() bumps the version even on a pure re-projection, so a
        # real revision is detected against the stored (pre-evaluation) record
        # combined with the baggage-side revision counters.
        connection = self.connections.evaluate(stored)

        inbound = self.connections.flight_arrivals.get(
            connection.inbound_flight, connection.inbound_onblock,
        )
        outbound = self.connections.flight_departures.get(
            connection.outbound_flight, connection.outbound_offblock,
        )
        required = self.connections.effective_minimum(connection)
        deadline = outbound  # passenger must be at the gate by off-block
        passenger_ready = inbound
        passenger_margin = int((deadline - passenger_ready).total_seconds() // 60) - required

        items: list[BagLineItem] = []
        for tag in self.bag_tags.get((passenger_id, outbound_flight), ()):
            bag = self.reconciler.bags[tag]
            if bag.state is BagState.LOADED:
                # Physically aboard the outbound aircraft: no cut-off remains.
                deadline: datetime | None = None
                feasible = True
            elif bag.state in _BAG_IN_FLOW_STATES:
                # Still moving: it must clear final handling before the
                # cut-off. This is baggage handling time, not the passenger's
                # gate-to-gate minimum (wheelchair assistance never slows a
                # container), but it does include the cross-terminal transfer.
                transfer = 0 if connection.same_terminal else self.connections.terminal_change_min
                cutoff = self.baggage_cutoff_min + transfer
                deadline = outbound - timedelta(minutes=cutoff)
                feasible = bag.last_seen_at <= deadline
            else:
                # MISSING or OFFLOADED: no achievable cut-off this flight.
                deadline = None
                feasible = False
            items.append(
                BagLineItem(
                    tag=tag,
                    state=bag.state,
                    last_seen_at=bag.last_seen_at,
                    ready_at=bag.last_seen_at,
                    deadline=deadline,
                    feasible=feasible,
                    revision=bag.revision,
                )
            )

        passenger_feasible = passenger_margin >= 0
        baggage_feasible = bool(items) and all(item.feasible for item in items)
        revised = stored.version > 1 or any(item.revision > 0 for item in items)

        if connection.state == ConnectionState.COMPLETED:
            joint_state = ConnectionState.COMPLETED
        elif passenger_feasible and baggage_feasible:
            joint_state = ConnectionState.PROTECTED if connection.protected_until is not None else ConnectionState.PLANNED
        else:
            joint_state = ConnectionState.MISCONNECTED

        return JointConnectionVerdict(
            passenger_id=passenger_id,
            outbound_flight=outbound_flight,
            connection=connection,
            passenger_ready_at=inbound,
            passenger_deadline=deadline,
            passenger_margin_min=passenger_margin,
            passenger_feasible=passenger_feasible,
            bags=tuple(items),
            revised=revised,
            joint_state=joint_state,
        )

    def replay(
        self,
        passenger_id: str,
        outbound_flight: str,
        *,
        late_scans: tuple[Scan, ...] = (),
        inbound_onblock: datetime | None = None,
        outbound_offblock: datetime | None = None,
    ) -> JointConnectionVerdict:
        """Ingest late facts in actual-occurrence order and re-derive both lines.

        The revision is applied to both timelines before any verdict is
        produced: callers can never observe a baggage projection that was
        rewritten by a late scan while the passenger projection (or vice
        versa) still reflects the old timeline.
        """
        key = self._key(passenger_id, outbound_flight)
        connection = self.connections.connections[key]
        ordered_facts = sorted(
            [("scan", utc(s.occurred_at), s) for s in late_scans]
            + (
                [("inbound", utc(inbound_onblock), inbound_onblock)]
                if inbound_onblock is not None else []
            )
            + (
                [("outbound", utc(outbound_offblock), outbound_offblock)]
                if outbound_offblock is not None else []
            ),
            key=lambda fact: fact[1],
        )

        for kind, _at, payload in ordered_facts:
            if kind == "scan":
                self.reconciler.ingest(payload)
            else:
                self.connections.update_flight(
                    connection.inbound_flight if kind == "inbound" else outbound_flight,
                    onblock=payload if kind == "inbound" else None,
                    offblock=payload if kind == "outbound" else None,
                )

        return self.evaluate(passenger_id, outbound_flight)
