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
    def __post_init__(self):object.__setattr__(self,'last_seen_at',utc(self.last_seen_at))

@dataclass(frozen=True)
class Scan:
    scan_id:str;tag:str;occurred_at:datetime;location:str;action:str;flight_id:str|None=None;container_id:str|None=None
    def __post_init__(self):object.__setattr__(self,'occurred_at',utc(self.occurred_at))

class BagReconciler:
    transitions={
        BagState.ACCEPTED:{'screen':BagState.SCREENED,'transfer':BagState.TRANSFER,'missing':BagState.MISSING},
        BagState.SCREENED:{'sort':BagState.SORTED,'missing':BagState.MISSING},
        BagState.SORTED:{'load':BagState.LOADED,'missing':BagState.MISSING},
        BagState.LOADED:{'offload':BagState.OFFLOADED,'missing':BagState.MISSING},
        BagState.OFFLOADED:{'sort':BagState.SORTED,'transfer':BagState.TRANSFER},
        BagState.TRANSFER:{'screen':BagState.SCREENED,'sort':BagState.SORTED,'missing':BagState.MISSING},
        BagState.MISSING:{'screen':BagState.SCREENED,'sort':BagState.SORTED,'load':BagState.LOADED},
    }

    def __init__(self,max_late_scan_minutes:int=30):
        self.max_late=timedelta(minutes=max_late_scan_minutes);self.bags:dict[str,Bag]={};self._seen_scans:set[str]=set();self._history:dict[str,list[Scan]]={};self._registered:dict[str,Bag]={}

    def register(self,bag:Bag)->None:
        if bag.tag in self.bags:raise ValueError('duplicate bag')
        self.bags[bag.tag]=bag;self._history[bag.tag]=[];self._registered[bag.tag]=bag

    def _replay(self,tag:str,scans:list[Scan])->Bag:
        bag=self._registered[tag]
        state=bag.state;last_seen=bag.last_seen_at;location=bag.location;flight_id=bag.flight_id;container_id=bag.container_id
        for scan in scans:
            nxt=self.transitions.get(state,{}).get(scan.action)
            if nxt is None:
                raise RuntimeError(f"invalid bag transition {state}:{scan.action}")
            state=nxt
            last_seen=max(last_seen,scan.occurred_at)
            location=scan.location
            flight_id=scan.flight_id or flight_id
            container_id=scan.container_id if scan.container_id is not None else container_id
        return replace(bag,state=state,last_seen_at=last_seen,location=location,flight_id=flight_id,container_id=container_id,sequence=bag.sequence+len(scans))

    def ingest(self, scan: Scan) -> Bag:
        if scan.scan_id in self._seen_scans:
            return self.bags[scan.tag]
        bag = self.bags[scan.tag]
        if scan.occurred_at < bag.last_seen_at - self.max_late:
            raise RuntimeError("scan outside late-arrival tolerance")
        ordered=sorted(self._history[scan.tag]+[scan],key=lambda s:s.occurred_at)
        updated=self._replay(scan.tag,ordered)
        self._history[scan.tag].append(scan)
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
