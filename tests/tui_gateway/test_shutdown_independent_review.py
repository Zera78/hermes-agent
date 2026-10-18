import queue
import threading
from unittest.mock import Mock
import pytest
from hermes_cli import backend_retirement as br
from tools import async_delegation as d
from tools.process_registry import process_registry
from tui_gateway import server

def seed():
    event={'type':'async_delegation','delegation_id':'review-only-d','session_key':'review-only-owner'}
    d._persist_dispatch(dict(event,goal='Synthetic review',dispatched_at=2_000_000_000.0))
    d._persist_completion(dict(event,status='completed'),{'status':'completed','summary':'Synthetic review'})
    return event

def row():
    with d._transaction() as db:
        return tuple(db.execute('SELECT delivery_state,delivery_attempts,delivery_claim,delivered_at FROM async_delegations WHERE delegation_id=?',('review-only-d',)).fetchone())

def test_stop_during_real_claim_preserves_pending_but_spends_attempt(monkeypatch):
    fence=br.RetirementFence(); monkeypatch.setattr(br,'retirement',fence)
    event=seed(); session={'history_lock':threading.RLock(),'running':True,'_notif_stop':threading.Event()}
    q=queue.Queue(); monkeypatch.setattr(process_registry,'completion_queue',q)
    claim=d.claim_event_delivery
    def then_stop(*args):
        token=claim(*args); fence.close_admission(); return token
    monkeypatch.setattr(d,'claim_event_delivery',then_stop)
    submit=Mock(); monkeypatch.setattr(server,'_notif_submit',submit)
    server._notif_dispatch_event('review-live',session,event,'Synthetic')
    assert row()==('pending',1,None,None)
    assert q.get_nowait()==event and not session['running']
    submit.assert_not_called(); fence.wait_quiescent(0)
    print('CLAIM_RACE: pending retained, original offer deferred, no ACK, one attempt retained')

def test_stop_after_submit_preserves_ack_and_cannot_replay(monkeypatch):
    fence=br.RetirementFence(); monkeypatch.setattr(br,'retirement',fence)
    event=seed(); session={'history_lock':threading.RLock(),'running':True,'_notif_stop':threading.Event()}
    def accepted_submit(*args, **kwargs):
        fence.close_admission()
        return True
    submit=Mock(side_effect=accepted_submit)
    monkeypatch.setattr(server,'_notif_submit',submit)
    server._notif_dispatch_event('review-live',session,event,'Synthetic')
    state,attempts,claim,delivered=row()
    assert state=='delivered' and attempts==1 and claim is None and delivered is not None
    assert d.restore_undelivered_completions(queue.Queue())==0
    submit.assert_called_once(); fence.wait_quiescent(0)
    print('POST_SUBMIT_CLOSE: ACK retained, no replay of accepted work')
