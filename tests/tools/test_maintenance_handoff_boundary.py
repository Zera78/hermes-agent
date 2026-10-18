"""A maintenance close racing the real durable dispatch must refuse its worker handoff."""
from contextlib import contextmanager
from unittest.mock import Mock
from hermes_cli import backend_retirement as br
from tools import async_delegation as delivery


def test_closed_worker_reservation_never_submits_or_underflows(monkeypatch):
    class Gate:
        allowed=True
        active=0
        def acquire(self):
            if not self.allowed: return False
            self.active+=1; return True
        def release(self): self.active-=1
        @contextmanager
        def work(self):
            admitted=self.acquire()
            try: yield admitted
            finally:
                if admitted: self.release()
    gate=Gate(); monkeypatch.setattr(br,'retirement',gate)
    persist=delivery._persist_dispatch
    def persist_then_close(record):
        persist(record); gate.allowed=False
    monkeypatch.setattr(delivery,'_persist_dispatch',persist_then_close)
    future=Mock(); future.add_done_callback.side_effect=lambda fn:fn(future)
    executor=Mock(); executor.submit.return_value=future
    monkeypatch.setattr(delivery,'_get_executor',lambda _:executor)
    result=delivery.dispatch_async_delegation(goal='Synthetic handoff',context=None,toolsets=None,
        role=None,model=None,session_key='synthetic-owner',runner=Mock())
    assert result['status']=='rejected'
    executor.submit.assert_not_called()
    assert gate.active==0
    with delivery._transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM async_delegations').fetchone()[0]==0
