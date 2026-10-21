"""Cancellation-helper publication, complete tail and owner profile contracts."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from agent import background_review as review, memory_provider
from hermes_cli import backend_retirement as br
from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override

@pytest.fixture
def fence(monkeypatch):
    fence=br.RetirementFence();monkeypatch.setattr(br,'retirement',fence);return fence

@pytest.mark.parametrize('raises',[False,True])
def test_cancellation_owns_complete_profile_bound_tail(fence,monkeypatch,tmp_path,raises):
    entered=threading.Event();release=threading.Event();finished=threading.Event();seen=[];threads=[]
    spawn=memory_provider.spawn_context_thread
    def capture(*args,**kwargs):
        worker=spawn(*args,**kwargs);threads.append(worker);return worker
    monkeypatch.setattr(memory_provider,'spawn_context_thread',capture)
    def interrupt(message=None):
        entered.set();assert release.wait(3);seen.append(get_hermes_home());finished.set()
        if raises:raise ValueError('synthetic abort tail failure')
    home=tmp_path/'secondary';home.mkdir();token=set_hermes_home_override(home)
    try:
        review._interrupt_background_review(SimpleNamespace(interrupt=interrupt))
    finally:reset_hermes_home_override(token)
    try:
        assert entered.wait(2);fence.close_admission()
        assert fence.active_count()==1
        with pytest.raises(TimeoutError):fence.wait_quiescent(0)
    finally:
        release.set();threads[0].join(3)
    assert finished.is_set() and not threads[0].is_alive();assert seen==[home]
    fence.wait_quiescent(0);assert fence.active_count()==0

@pytest.mark.parametrize('phase',['construct','start'])
def test_definite_cancel_helper_start_failure_releases_exact_reservation(fence,monkeypatch,phase):
    class Failed:
        ident=None
        def start(self):raise RuntimeError('synthetic definite start failure')
    def spawn(*args,**kwargs):
        if phase=='construct':raise RuntimeError('synthetic construction failure')
        return Failed()
    monkeypatch.setattr(memory_provider,'spawn_context_thread',spawn)
    fork=SimpleNamespace(interrupt=Mock())
    review._interrupt_background_review(fork)
    fence.close_admission();fence.wait_quiescent(0);fork.interrupt.assert_not_called()
    assert fence.active_count()==0

def test_uncertain_cancel_helper_start_keeps_refusal(fence,monkeypatch):
    class Unknown:
        ident=None
        def start(self):raise KeyboardInterrupt('synthetic unknown start; no signal')
    monkeypatch.setattr(memory_provider,'spawn_context_thread',lambda *args,**kwargs:Unknown())
    with pytest.raises(KeyboardInterrupt):review._interrupt_background_review(SimpleNamespace())
    fence.close_admission();assert fence.active_count()==1
    with pytest.raises(TimeoutError):fence.wait_quiescent(0)
    # No actual thread exists. Its isolated unknown fence is disposed, never proved.

def test_start_exception_after_publication_cannot_drop_running_tail(fence,monkeypatch):
    entered=threading.Event();release=threading.Event();threads=[]
    def interrupt(message=None):entered.set();assert release.wait(3)
    class Published:
        def __init__(self,target,**kwargs):
            self.worker=threading.Thread(target=target,name='synthetic-cancel');threads.append(self.worker)
        @property
        def ident(self):return self.worker.ident
        def start(self):
            self.worker.start();assert entered.wait(2)
            raise RuntimeError('synthetic exception after started publication')
    monkeypatch.setattr(memory_provider,'spawn_context_thread',lambda target,**kwargs:Published(target,**kwargs))
    review._interrupt_background_review(SimpleNamespace(interrupt=interrupt))
    try:
        fence.close_admission();assert fence.active_count()==1
        with pytest.raises(TimeoutError):fence.wait_quiescent(0)
    finally:release.set();threads[0].join(3)
    fence.wait_quiescent(0);assert fence.active_count()==0

def test_admitted_parent_can_transfer_cancel_tail_after_close(fence,monkeypatch):
    entered=threading.Event();release=threading.Event();threads=[]
    spawn=memory_provider.spawn_context_thread
    def capture(*args,**kwargs):
        thread=spawn(*args,**kwargs);threads.append(thread);return thread
    monkeypatch.setattr(memory_provider,'spawn_context_thread',capture)
    def interrupt(message=None):entered.set();assert release.wait(3)
    with fence.work() as admitted:
        assert admitted;fence.close_admission()
        review._interrupt_background_review(SimpleNamespace(interrupt=interrupt))
        assert entered.wait(2);assert fence.active_count()==2
    try:
        assert fence.active_count()==1
        with pytest.raises(TimeoutError):fence.wait_quiescent(0)
    finally:release.set();threads[0].join(3)
    fence.wait_quiescent(0);assert fence.reserve() is None

def test_closed_root_refuses_before_cancel_helper_publication(fence,monkeypatch):
    spawn=Mock(side_effect=AssertionError('unowned cancellation publication'))
    monkeypatch.setattr(memory_provider,'spawn_context_thread',spawn)
    fence.close_admission();review._interrupt_background_review(SimpleNamespace())
    spawn.assert_not_called();fence.wait_quiescent(0)
