"""Synthetic maintenance boundaries execute the rebound native code, without workers/models."""
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tui_gateway import server
from hermes_cli import backend_retirement as br
from tools import async_delegation as delivery
from tools.process_registry import process_registry


@pytest.fixture
def fence(monkeypatch):
    value = br.RetirementFence()
    monkeypatch.setattr(br, 'retirement', value)
    return value


def session():
    return {'history_lock': threading.RLock(), 'running': False,
            'session_key': 'synthetic-owner', '_notif_stop': threading.Event()}


def event():
    return {'type': 'async_delegation', 'delegation_id': 'synthetic-completion',
            'session_key': 'synthetic-owner', 'origin_ui_session_id': 'synthetic-live'}


def test_stop_before_start_does_not_restore_or_drain(monkeypatch, fence):
    s = session(); s['_notif_stop'].set()
    q = queue.Queue(); q.put(event())
    restore = Mock(); dispatch = Mock()
    monkeypatch.setattr(process_registry, 'completion_queue', q)
    monkeypatch.setattr(process_registry, 'restore_completions', restore)
    monkeypatch.setattr(server, '_notif_handle_ready', dispatch)
    server._notification_poller_scoped_loop(s['_notif_stop'], 'synthetic-live', s)
    restore.assert_not_called(); dispatch.assert_not_called()
    assert q.get_nowait() == event() and q.empty()


def test_stop_during_queue_get_defers_without_dispatch(monkeypatch, fence):
    s = session()
    class StopQueue(queue.Queue):
        def get(self, *a, **k):
            value = super().get(*a, **k)
            s['_notif_stop'].set()
            return value
    q = StopQueue(); q.put(event())
    claim = Mock(return_value='claim'); submit = Mock()
    monkeypatch.setattr(process_registry, 'completion_queue', q)
    monkeypatch.setattr(process_registry, 'restore_completions', lambda: None)
    monkeypatch.setattr(delivery, 'maybe_sweep_orphaned_completions', lambda q: None)
    for name in ('_maybe_fire_tui_loop_tick', '_maybe_fire_tui_heartbeat_tick', '_notif_poll_kanban'):
        monkeypatch.setattr(server, name, lambda *a: None)
    monkeypatch.setattr(server, '_session_owns_notification_event', lambda *a: True)
    monkeypatch.setattr(server, '_notification_event_belongs_elsewhere', lambda *a: False)
    monkeypatch.setattr(delivery, 'claim_event_delivery', claim)
    monkeypatch.setattr(server, '_notif_submit', submit)
    monkeypatch.setattr(server, '_emit', lambda *a: None)
    server._notification_poller_scoped_loop(s['_notif_stop'], 'synthetic-live', s)
    claim.assert_not_called(); submit.assert_not_called()
    assert q.qsize() == 1


def test_stop_before_durable_claim_keeps_complete_row_identical(monkeypatch, fence):
    # Real fixture SQLite persistence and native claim/ACK seams, all under runner's synthetic HOME.
    payload = event()
    delivery._persist_dispatch(dict(payload, goal='Synthetic', dispatched_at=2_000_000_000.0))
    delivery._persist_completion(dict(payload, status='completed'), {'status':'completed','summary':'Synthetic'})
    with delivery._transaction() as db:
        before = tuple(db.execute('SELECT rowid,* FROM async_delegations WHERE delegation_id=?', (payload['delegation_id'],)).fetchone())
    s = session(); s['running'] = True; s['_notif_stop'].set()
    submit = Mock(); monkeypatch.setattr(server, '_notif_submit', submit)
    q = queue.Queue(); monkeypatch.setattr(process_registry, 'completion_queue', q)
    server._notif_dispatch_event('synthetic-live', s, payload, 'Synthetic')
    with delivery._transaction() as db:
        after = tuple(db.execute('SELECT rowid,* FROM async_delegations WHERE delegation_id=?', (payload['delegation_id'],)).fetchone())
    assert after == before
    submit.assert_not_called()
    assert not s['running'] and q.qsize() == 1


def test_stop_during_claim_releases_and_defers_without_ack(monkeypatch, fence):
    s = session(); s['running'] = True
    def claim(*a):
        s['_notif_stop'].set()
        return 'synthetic-claim'
    released = Mock(); ack = Mock(); submit = Mock()
    q = queue.Queue(); monkeypatch.setattr(process_registry, 'completion_queue', q)
    monkeypatch.setattr(delivery, 'claim_event_delivery', claim)
    monkeypatch.setattr(delivery, 'release_event_delivery', released)
    monkeypatch.setattr(delivery, 'complete_event_delivery', ack)
    monkeypatch.setattr(server, '_notif_submit', submit)
    server._notif_dispatch_event('synthetic-live', s, event(), 'Synthetic')
    submit.assert_not_called(); ack.assert_not_called()
    released.assert_called_once_with(event(), 'synthetic-claim')
    assert not s['running'] and q.qsize() == 1


def test_closed_admission_defers_batch_without_claim(monkeypatch, fence):
    fence.close_admission()
    s = session(); q = queue.Queue(); claim = Mock()
    monkeypatch.setattr(delivery, 'claim_event_delivery', claim)
    events = [({'type':'completion','session_id':'synthetic-process'}, 'Synthetic')]
    server._notif_dispatch_completions('synthetic-live', s, events, SimpleNamespace(completion_queue=q), None)
    claim.assert_not_called()
    assert q.qsize() == 1 and not s['running']


def test_fence_bounds_existing_work_and_never_reopens(fence):
    assert fence.acquire()
    fence.close_admission(); fence.close_admission()
    assert not fence.acquire()
    with pytest.raises(TimeoutError): fence.wait_quiescent(0)
    fence.release()
    fence.wait_quiescent(0)
    assert not fence.acquire()


def test_finalize_refuses_storage_cleanup_while_poller_alive(monkeypatch, fence):
    s = session(); alive = threading.Event(); release = threading.Event()
    t = threading.Thread(target=lambda: (alive.set(), release.wait(5)))
    t.start(); assert alive.wait(2)
    s['_notif_thread'] = t
    monkeypatch.setattr(server, '_NOTIFICATION_STOP_TIMEOUT', 0)
    cleanup = Mock(); monkeypatch.setattr(server, '_lock_vault_managers', cleanup)
    try:
        with pytest.raises(TimeoutError): server._finalize_session(s)
        assert not s.get('_finalized')
        cleanup.assert_not_called()
    finally:
        release.set(); t.join(2)
    assert not t.is_alive()


def test_normal_delivery_is_acknowledged_once_and_not_replayed(monkeypatch, fence):
    payload = event()
    delivery._persist_dispatch(dict(payload, goal='Synthetic', dispatched_at=2_000_000_000.0))
    delivery._persist_completion(dict(payload, status='completed'), {'status':'completed','summary':'Synthetic'})
    s = session(); s['running'] = True
    submit = Mock(); monkeypatch.setattr(server, '_notif_submit', submit)
    server._notif_dispatch_event('synthetic-live', s, payload, 'Synthetic')
    s['running'] = True
    server._notif_dispatch_event('synthetic-live', s, payload, 'Synthetic duplicate')
    assert submit.call_count == 1
    with delivery._transaction() as db:
        row = db.execute('SELECT delivery_state FROM async_delegations WHERE delegation_id=?',(payload['delegation_id'],)).fetchone()
    assert row[0] == 'delivered'
    offers=queue.Queue()
    assert delivery.restore_undelivered_completions(offers) == 0 and offers.empty()


def test_stop_during_batch_claim_does_not_claim_second_or_ack(monkeypatch, fence):
    s=session(); q=queue.Queue(); calls=[]; release=Mock(); ack=Mock(); submit=Mock()
    events=[({'type':'completion','session_id':name},'Synthetic') for name in ('one','two')]
    def claim(evt, consumer):
        calls.append(evt['session_id']); s['_notif_stop'].set(); return 'synthetic-claim'
    monkeypatch.setattr(delivery,'claim_event_delivery',claim)
    monkeypatch.setattr(delivery,'release_event_delivery',release)
    monkeypatch.setattr(delivery,'complete_event_delivery',ack)
    monkeypatch.setattr(server,'_notif_submit',submit)
    server._notif_dispatch_completions('synthetic-live',s,events,SimpleNamespace(completion_queue=q),None)
    assert calls==['one'] and q.qsize()==2 and not s['running']
    release.assert_called_once(); ack.assert_not_called(); submit.assert_not_called()


def test_closed_ready_snapshot_preserves_order_without_formatting_or_emission(monkeypatch, fence):
    fence.close_admission(); s=session(); q=queue.Queue(); fmt=Mock(); emit=Mock()
    monkeypatch.setattr(server,'_emit',emit)
    events=[event(),dict(event(),delegation_id='second')]
    emitted=set()
    server._notif_handle_ready('synthetic-live',s,events,emitted,SimpleNamespace(completion_queue=q),fmt,None)
    assert [q.get_nowait(),q.get_nowait()]==events and q.empty() and not emitted
    fmt.assert_not_called(); emit.assert_not_called()


def test_stop_after_admission_before_claim_defers(monkeypatch, fence):
    original=fence.reserve
    def stopping_reserve():
        admitted=original(); fence.close_admission(); return admitted
    monkeypatch.setattr(fence,'reserve',stopping_reserve)
    s=session(); s['running']=True; q=queue.Queue(); claim=Mock()
    monkeypatch.setattr(process_registry,'completion_queue',q)
    monkeypatch.setattr(delivery,'claim_event_delivery',claim)
    server._notif_dispatch_event('synthetic-live',s,event(),'Synthetic')
    claim.assert_not_called(); assert q.qsize()==1 and not s['running']
    fence.wait_quiescent(0)


def test_explicit_submit_refusal_does_not_acknowledge(monkeypatch,fence):
    s=session(); s['running']=True; release=Mock(); ack=Mock(); defer=Mock()
    monkeypatch.setattr(delivery,'claim_event_delivery',lambda *a:'synthetic-claim')
    monkeypatch.setattr(delivery,'release_event_delivery',release)
    monkeypatch.setattr(delivery,'complete_event_delivery',ack)
    monkeypatch.setattr(server,'_notif_defer_event',defer)
    monkeypatch.setattr(server,'_notif_refusal_backoff',lambda *a:None)
    monkeypatch.setattr(server,'_run_prompt_submit',lambda *a,**k:False)
    monkeypatch.setattr(server,'_emit',lambda *a,**k:None)
    server._notif_dispatch_event('synthetic-live',s,event(),'Synthetic')
    ack.assert_not_called(); release.assert_not_called()
    defer.assert_called_once_with(event(), 'synthetic-claim'); assert not s['running']


def test_start_after_close_never_spawns_poller_or_worker(monkeypatch,fence):
    fence.close_admission(); target=Mock(); s=session()
    monkeypatch.setattr(server,'_wire_desktop_sinks',Mock(side_effect=AssertionError('Must not spawn')))
    assert server._start_notification_poller('synthetic-live',s).is_set()
    assert server._start_session_work(target,name='synthetic-never-started',session=s) is None
    target.assert_not_called()


def test_worker_exception_releases_admission(monkeypatch,fence):
    failed=threading.Event()
    def hook(args): failed.set()
    monkeypatch.setattr(threading,'excepthook',hook)
    def target(): raise RuntimeError('Synthetic worker failure')
    thread=server._start_session_work(target,name='synthetic-failure')
    thread.join(2); assert failed.wait(2) and not thread.is_alive()
    fence.close_admission(); fence.wait_quiescent(0)


def test_stop_does_not_treat_user_cancel_as_maintenance(fence):
    s=session(); s['_turn_cancel_requested']=True
    assert not server._notif_claim_turn(s)
    assert not fence.is_closed() and not s['_notif_stop'].is_set()
    s['_turn_cancel_requested']=False
    assert server._notif_claim_turn(s)


def test_async_handoff_refusal_does_not_spawn_or_underflow(monkeypatch,fence):
    original=delivery._persist_dispatch
    def persist_then_stop(record):
        original(record); fence.close_admission()
    monkeypatch.setattr(delivery,'_persist_dispatch',persist_then_stop)
    executor=Mock(); monkeypatch.setattr(delivery,'_get_executor',lambda _:executor)
    handle=delivery.dispatch_async_delegation(goal='Synthetic',context=None,toolsets=None,role=None,model=None,
        session_key='synthetic-owner',runner=Mock(side_effect=AssertionError('No child may execute')))
    assert handle['status']=='rejected'
    executor.submit.assert_not_called()
    assert fence.active_count()==0
    with delivery._transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM async_delegations').fetchone()[0]==0


def test_closed_builder_wakes_waiters_without_constructing_agent(monkeypatch,fence):
    s=session(); s['agent_ready']=threading.Event()
    make=Mock(side_effect=AssertionError('No model construction'))
    monkeypatch.setattr(server,'_make_agent',make)
    fence.close_admission()
    server._start_agent_build('synthetic-live',s)
    assert s['agent_ready'].is_set() and s['_agent_build_thread'] is None
    make.assert_not_called()


def test_closed_resume_hydration_closes_unused_handle_without_read(monkeypatch,fence):
    db=Mock(); fence.close_admission()
    server._schedule_resume_hydration('synthetic-live','synthetic-owner',db,close_db=True)
    db.close.assert_called_once(); db.retire_undrained_queue_rows.assert_not_called()


def test_reaper_timer_cannot_write_after_maintenance_close(monkeypatch,fence):
    callbacks=[]; work=Mock()
    class Timer:
        def __init__(self,delay,target): self.target=target
        def start(self): callbacks.append(self.target)
    monkeypatch.setattr(threading,'Timer',Timer)
    server._reaper_daemon_timer(0,work,'Synthetic')
    fence.close_admission(); callbacks[0]()
    work.assert_not_called(); fence.wait_quiescent(0)


def test_normal_prepare_cancel_commit_contract_is_preserved(monkeypatch,fence):
    monkeypatch.setattr('hermes_cli.web_server_idle_proof.idle_proof',lambda:{'idle':True})
    prepared=fence.prepare(); assert prepared['ok']
    assert not fence.acquire()
    assert fence.cancel(prepared['token'])=={'ok':True}
    assert fence.acquire(); fence.release()
    prepared=fence.prepare()
    assert fence.commit(prepared['token'])=={'ok':True}
    assert fence.commit(prepared['token'])=={'ok':True}
    assert fence.prepare()['token']==prepared['token']
    assert not fence.acquire()


def test_pollers_stop_before_profile_scope_is_opened(monkeypatch,fence):
    s=session(); s['_notif_stop'].set()
    scope=Mock(side_effect=AssertionError('No profile hydration after stop'))
    monkeypatch.setattr(server,'_session_profile_runtime_scope',scope)
    server._notification_poller_loop(s['_notif_stop'],'synthetic-live',s)
    scope.assert_not_called()
