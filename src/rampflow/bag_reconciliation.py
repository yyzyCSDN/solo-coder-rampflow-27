from __future__ import annotations
from dataclasses import dataclass,replace
from datetime import datetime,timedelta
from enum import Enum
from typing import Iterable,Mapping
from .timeutil import utc

class BagState(str,Enum):
    ACCEPTED='accepted';SCREENED='screened';SORTED='sorted';LOADED='loaded';OFFLOADED='offloaded';TRANSFER='transfer';MISSING='missing'

@dataclass(frozen=True)
class Bag:
    tag:str
    passenger_id:str
    flight_id:str
    state:BagState
    last_seen_at:datetime
    location:str
    container_id:str|None=None
    sequence:int=0
    revision:int=0
    def __post_init__(self):object.__setattr__(self,'last_seen_at',utc(self.last_seen_at))

@dataclass(frozen=True)
class Scan:
    scan_id:str;tag:str;occurred_at:datetime;location:str;action:str;flight_id:str|None=None;container_id:str|None=None
    def __post_init__(self):object.__setattr__(self,'occurred_at',utc(self.occurred_at))

# Allowed transitions, evaluated against the state implied by the timeline
# position of a scan — never against the state at ingestion time.
_ALLOWED = {
    BagState.ACCEPTED:{'screen':BagState.SCREENED,'transfer':BagState.TRANSFER,'missing':BagState.MISSING},
    BagState.SCREENED:{'sort':BagState.SORTED,'missing':BagState.MISSING},
    BagState.SORTED:{'load':BagState.LOADED,'missing':BagState.MISSING},
    BagState.LOADED:{'offload':BagState.OFFLOADED,'missing':BagState.MISSING},
    BagState.OFFLOADED:{'sort':BagState.SORTED,'transfer':BagState.TRANSFER},
    BagState.TRANSFER:{'screen':BagState.SCREENED,'sort':BagState.SORTED,'missing':BagState.MISSING},
    BagState.MISSING:{'screen':BagState.SCREENED,'sort':BagState.SORTED,'load':BagState.LOADED},
}

class BagReconciler:
    def __init__(self,max_late_scan_minutes:int=30):
        self.max_late=timedelta(minutes=max_late_scan_minutes);self.bags:dict[str,Bag]={};self._seen_scans:set[str]=set();self._history:dict[str,list[Scan]]={};self._registered:dict[str,Bag]={}

    def register(self,bag:Bag)->None:
        if bag.tag in self.bags:raise ValueError('duplicate bag')
        self.bags[bag.tag]=bag;self._history[bag.tag]=[];self._registered[bag.tag]=bag

    def _ordered(self,tag:str)->list[Scan]:
        # Stable sort: scans reporting the same physical time keep the order in
        # which the system learned about them.
        return sorted(self._history[tag],key=lambda s:s.occurred_at)

    def history(self,tag:str)->tuple[Scan,...]:
        """Scans for a tag in actual-occurrence order, regardless of ingestion."""
        return tuple(self._ordered(tag))

    def _replay(self,tag:str,revision:int)->Bag:
        # The registered bag anchors the timeline; rebuild its projection by
        # replaying every known scan from the beginning, in real-world order,
        # so a late-arriving scan revises history rather than being applied on
        # top of (and regressing) the current projection.
        registered=self._registered[tag]
        state=registered.state
        flight_id=registered.flight_id
        container_id=registered.container_id
        location=registered.location
        last_seen=registered.last_seen_at
        bag=registered
        for sequence,scan in enumerate(self._ordered(tag),start=1):
            nxt=_ALLOWED.get(state,{}).get(scan.action)
            if nxt is None:
                raise RuntimeError(f"invalid bag transition {state}:{scan.action}")
            state=nxt
            last_seen=max(last_seen,scan.occurred_at)
            location=scan.location
            flight_id=scan.flight_id or flight_id
            if scan.container_id is not None:container_id=scan.container_id
            bag=replace(registered,state=state,last_seen_at=last_seen,location=location,flight_id=flight_id,container_id=container_id,sequence=sequence,revision=revision)
        if not self._history[tag]:
            bag=replace(registered,revision=revision)
        return bag

    def ingest(self, scan: Scan) -> Bag:
        if scan.scan_id in self._seen_scans:
            return self.bags[scan.tag]
        bag = self.bags[scan.tag]
        latest=max((s.occurred_at for s in self._history[scan.tag]),default=bag.last_seen_at)
        if scan.occurred_at < latest - self.max_late:
            raise RuntimeError("scan outside late-arrival tolerance")
        self._history[scan.tag].append(scan)
        # A scan is a revision only when it reports something earlier than the
        # newest fact already on record; it must rewrite, not append to, state.
        revision=bag.revision+(1 if scan.occurred_at<latest else 0)
        try:
            updated=self._replay(scan.tag,revision)
        except Exception:
            # A rejected revision must leave the prior projection intact.
            self._history[scan.tag].pop()
            raise
        self._seen_scans.add(scan.scan_id)
        self.bags[scan.tag] = updated
        return updated

    def reconcile_departure(self,flight_id:str,boarded_passengers:Iterable[str])->Mapping[str,tuple[str,...]]:
        boarded=set(boarded_passengers);loaded=[];must_offload=[];missing=[]
        for bag in self.bags.values():
            if bag.flight_id!=flight_id:continue
            if bag.state==BagState.LOADED:
                loaded.append(bag.tag)
                if bag.passenger_id not in boarded:must_offload.append(bag.tag)
            elif bag.passenger_id in boarded and bag.state not in {BagState.OFFLOADED}:missing.append(bag.tag)
        return {'loaded':tuple(sorted(loaded)),'must_offload':tuple(sorted(must_offload)),'missing_for_boarded':tuple(sorted(missing))}

    def transfer_deadline(self,tag:str,departure:datetime,minimum_connect_min:int)->datetime:
        bag=self.bags[tag]
        departure=utc(departure)
        if bag.state not in {BagState.TRANSFER,BagState.SCREENED,BagState.SORTED}:raise RuntimeError('bag not in transfer flow')
        return departure-timedelta(minutes=minimum_connect_min)
