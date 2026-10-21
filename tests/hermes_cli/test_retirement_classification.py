"""Unsupported lifecycle classification dominates generic registration and cached proof."""
import threading
import pytest
from agent.memory_lifecycle import retain,retain_provider
from hermes_cli import backend_retirement as br

@pytest.fixture
def fence(monkeypatch):
    fence=br.RetirementFence();monkeypatch.setattr(br,'retirement',fence);return fence

class DuckProvider:
    def __init__(self):self.calls=0
    def maintenance_drain(self,deadline):self.calls+=1;return True

@pytest.mark.parametrize('first',['generic','unsupported'])
@pytest.mark.parametrize('repeats',[1,3])
def test_unsupported_classification_is_persistent_in_both_orders(fence,first,repeats):
    owner=DuckProvider()
    operations=[retain,retain_provider] if first=='generic' else [retain_provider,retain]
    for operation in operations:operation(owner)
    for _ in range(repeats):
        retain(owner);retain_provider(owner);retain(owner)
    fence.close_admission()
    for _ in range(2):
        with pytest.raises(RuntimeError,match='Unsupported memory lifecycle: DuckProvider'):
            fence.drain_owned(1)
        retain(owner)  # A later generic attempt cannot restore support on retry.
    assert owner.calls==0

def test_unsupported_classification_revokes_prior_cached_settlement(fence):
    owner=DuckProvider();retain(owner);fence.close_admission()
    fence.drain_owned(1);fence.drain_owned(1);assert owner.calls==1
    retain_provider(owner);retain(owner);retain_provider(owner)
    with pytest.raises(RuntimeError,match='Unsupported memory lifecycle: DuckProvider'):
        fence.drain_owned(1)
    assert owner.calls==1

def test_later_proof_cannot_reclassify_an_already_checked_owner_in_same_drain(fence):
    owner=DuckProvider();retain(owner)
    class Classifier:
        def maintenance_drain(self,deadline):
            retain_provider(owner);retain(owner);return True
    retain(Classifier());fence.close_admission()
    with pytest.raises(RuntimeError,match='Unsupported memory lifecycle: DuckProvider'):
        fence.drain_owned(1)
    assert owner.calls==1

def test_inflight_generic_success_cannot_erase_concurrent_unsupported_classification(fence):
    entered=threading.Event();release=threading.Event();errors=[]
    class Blocked(DuckProvider):
        def maintenance_drain(self,deadline):
            self.calls+=1;entered.set();assert release.wait(3);return True
    owner=Blocked();retain(owner);fence.close_admission()
    def drain():
        try:fence.drain_owned(2)
        except BaseException as error:errors.append(error)
    worker=threading.Thread(target=drain);worker.start()
    try:
        assert entered.wait(2);retain_provider(owner);retain(owner)
    finally:release.set();worker.join(3)
    assert not worker.is_alive();assert len(errors)==1
    assert isinstance(errors[0],RuntimeError) and 'Unsupported memory lifecycle: Blocked' in str(errors[0])
    assert owner.calls==1

def test_repeated_positive_registration_keeps_completed_cleanup_cached(fence):
    owner=DuckProvider();retain(owner);fence.close_admission()
    fence.drain_owned(1)
    for _ in range(3):retain(owner);fence.drain_owned(1)
    assert owner.calls==1
