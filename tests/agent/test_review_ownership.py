"""Admission linearization, complete ownership, and deferred handoffs at real source paths."""
import contextvars
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from hermes_cli import backend_retirement as br
from agent import review_idle_queue as idle, background_review as review
from run_agent import AIAgent


@pytest.fixture
def fence(monkeypatch):
    f = br.RetirementFence()
    monkeypatch.setattr(br, 'retirement', f)
    return f


@pytest.fixture
def queue(monkeypatch, fence):
    q = idle.ReviewIdleQueue()
    monkeypatch.setattr(q, '_ensure_thread', lambda: None)
    monkeypatch.setattr(q, '_server_idle', Mock(side_effect=AssertionError('no idle probe')))
    monkeypatch.setattr(idle, 'QUEUE', q)
    monkeypatch.setattr(review, 'load_background_review_settings', lambda: (True, {}))
    return q


def test_child_handoff_and_stale_context_cannot_reopen(fence):
    with fence.work() as admitted:
        assert admitted
        captured = contextvars.copy_context()
        fence.close_admission()
        child = fence.reserve()
        assert child is not None and fence.active_count() == 2
        with pytest.raises(RuntimeError, match='own settlement'):
            fence.wait_quiescent(0)
    assert fence.active_count() == 1
    assert captured.run(fence.reserve) is None
    with pytest.raises(TimeoutError): fence.wait_quiescent(0)
    child.finish(); child.finish()
    fence.wait_quiescent(0)
    with pytest.raises(RuntimeError): fence.release()
    assert fence.active_count() == 0


def test_enqueue_close_race_reserves_before_publication(queue, fence, monkeypatch):
    entered=threading.Event(); release=threading.Event(); errors=[]
    def now(): entered.set(); assert release.wait(3); return 0
    monkeypatch.setattr(queue, '_now', now)
    agent=SimpleNamespace(_spawn_background_review_now=Mock())
    def enqueue():
        try: queue.enqueue(agent, 'synthetic', {})
        except BaseException as e: errors.append(e)
    worker=threading.Thread(target=enqueue); worker.start()
    try:
        assert entered.wait(2)
        fence.close_admission()
        assert fence.active_count() == 1
    finally:
        release.set(); worker.join(3)
    assert not worker.is_alive() and not errors
    item=queue._pop_dispatchable()
    assert fence.active_count() == 1
    queue._dispatch(item)
    agent._spawn_background_review_now.assert_called_once()
    assert fence.active_count() == 0
    queue.enqueue(agent, 'refused', {})
    assert queue.pending_count() == 0


def test_popped_item_remains_owned_across_close_and_child_start(queue, fence, monkeypatch):
    popped=threading.Event(); release=threading.Event(); errors=[]
    def spawn(**kw):
        child=fence.reserve()
        assert child is not None and fence.active_count() == 2
        child.finish()
    agent=SimpleNamespace(_spawn_background_review_now=spawn)
    queue.enqueue(agent,'synthetic',{})
    enqueued = queue._pending['synthetic'].enqueued_at
    monkeypatch.setattr(queue, '_now', lambda: enqueued + 2000)
    def dispatch():
        try:
            item=queue._pop_dispatchable()
            popped.set(); assert release.wait(3)
            queue._dispatch(item)
        except BaseException as e: errors.append(e)
    worker=threading.Thread(target=dispatch);worker.start()
    try:
        assert popped.wait(2)
        assert queue.pending_count() == 0 and fence.active_count() == 1
        fence.close_admission()
    finally:
        release.set();worker.join(3)
    assert not worker.is_alive() and not errors
    fence.wait_quiescent(0)


def test_coalesce_and_disabled_dispatch_settle_exactly_once(queue, fence, monkeypatch):
    agent=SimpleNamespace(_spawn_background_review_now=Mock())
    queue.enqueue(agent,'synthetic',{'snapshot':1})
    queue.enqueue(agent,'synthetic',{'snapshot':2})
    assert fence.active_count() == 1
    fence.close_admission()
    monkeypatch.setattr(queue,'_still_enabled',lambda item:False)
    queue._dispatch(queue._pop_dispatchable())
    assert fence.active_count() == 0
    agent._spawn_background_review_now.assert_not_called()


def test_dispatch_exception_and_start_failure_settle(queue, fence, monkeypatch):
    agent=SimpleNamespace(_spawn_background_review_now=Mock(side_effect=ValueError('synthetic')))
    queue.enqueue(agent,'synthetic',{})
    fence.close_admission()
    with pytest.raises(ValueError): queue._dispatch(queue._pop_dispatchable())
    assert fence.active_count() == 0


@pytest.mark.parametrize('phase',['publication','dispatcher'])
def test_queue_thread_start_failure_removes_owned_publication(queue, fence, monkeypatch, phase):
    def fail(): raise RuntimeError('synthetic start failure')
    monkeypatch.setattr(queue, '_now' if phase=='publication' else '_ensure_thread', fail)
    with pytest.raises(RuntimeError): queue.enqueue(object(),'synthetic',{})
    assert queue.pending_count() == 0 and fence.active_count() == 0


@pytest.mark.parametrize('failure', ['thread', 'prepare', 'early_return', 'target'])
def test_review_start_and_early_exit_release_full_lifetime(failure, fence, monkeypatch):
    import run_agent
    class ImmediateThread:
        def __init__(self, *, target, **kwargs): self.target=target
        def start(self): self.target()
    agent=object.__new__(AIAgent)
    agent._background_review_run=None
    agent._background_review_lock=threading.Lock()
    monkeypatch.setattr(run_agent.threading,'Thread',ImmediateThread)
    if failure != 'early_return':
        monkeypatch.setattr(review, '_run_review_in_thread', lambda *a,**k:None)
    if failure == 'thread':
        class BadThread(ImmediateThread):
            def start(self): raise RuntimeError('start failure')
        monkeypatch.setattr(run_agent.threading,'Thread',BadThread)
    elif failure == 'prepare':
        monkeypatch.setattr(review,'spawn_background_review_thread',Mock(side_effect=ValueError('prepare failure')))
    elif failure == 'early_return':
        # Actual capability early return used to leave the request token installed.
        monkeypatch.setattr(run_agent.threading,'Thread',ImmediateThread)
        monkeypatch.setattr(review,'_parent_can_emit_tool_calls',lambda a:False)
        monkeypatch.setattr(review,'_resolve_review_runtime',lambda *a:{})
    else:
        monkeypatch.setattr(review,'_run_review_in_thread',Mock(side_effect=ValueError('target failure')))
    if failure in ('thread','prepare','target'):
        with pytest.raises((RuntimeError,ValueError)):
            agent._spawn_background_review_now([],review_memory=True,task_cfg={})
    else:
        agent._spawn_background_review_now([],review_memory=True,task_cfg={})
    assert fence.active_count() == 0 and agent._background_review_run is None


def test_interrupted_thread_start_keeps_unknown_ownership(fence, monkeypatch):
    import run_agent
    class UnknownStart:
        def __init__(self, **kwargs): pass
        def start(self): raise KeyboardInterrupt('synthetic uncertain start; no process signal')
    agent=object.__new__(AIAgent)
    agent._background_review_run=None
    agent._background_review_lock=threading.Lock()
    monkeypatch.setattr(run_agent.threading,'Thread',UnknownStart)
    with pytest.raises(KeyboardInterrupt):
        agent._spawn_background_review_now([],review_memory=True,task_cfg={})
    fence.close_admission()
    assert fence.active_count() == 1
    with pytest.raises(TimeoutError): fence.wait_quiescent(0)
    assert fence.reserve() is None
