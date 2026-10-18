"""Admission, fallback, initialization and explicit extension boundaries, offline."""
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from agent.memory_manager import MemoryManager
from hermes_cli import backend_retirement as br


@pytest.fixture
def fence(monkeypatch):
    f = br.RetirementFence(); monkeypatch.setattr(br, 'retirement', f)
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: {})
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: {})
    return f


def test_inline_fallback_owns_complete_execution(fence, monkeypatch):
    import tools.daemon_pool as pool
    def fail(**kwargs): raise RuntimeError('synthetic executor allocation failure')
    monkeypatch.setattr(pool, 'DaemonThreadPoolExecutor', fail)
    manager = MemoryManager(); entered = threading.Event(); release = threading.Event()
    def write(): entered.set(); assert release.wait(3)
    worker = threading.Thread(target=lambda: manager._submit_background(write)); worker.start()
    try:
        assert entered.wait(2); fence.close_admission()
        with pytest.raises(TimeoutError): fence.wait_quiescent(0)
    finally:
        release.set(); worker.join(3)
    assert not worker.is_alive(); fence.wait_quiescent(0)


def test_submit_failure_after_enqueue_does_not_duplicate_inline_write(fence):
    queued = []; writes = []
    class Executor:
        def submit(self, fn): queued.append(fn); raise RuntimeError('synthetic worker start failure')
    manager = MemoryManager(); manager._sync_executor = Executor()
    manager._submit_background(lambda: writes.append('once'))
    assert writes == ['once']
    queued.pop()()
    assert writes == ['once']; assert fence.active_count() == 0


def test_close_race_reserves_before_executor_publication(fence):
    entered = threading.Event(); release = threading.Event(); writes = []
    class Executor:
        def submit(self, fn):
            entered.set(); assert release.wait(3)
            from concurrent.futures import Future
            future = Future(); fn(); future.set_result(None); return future
    manager = MemoryManager(); manager._sync_executor = Executor()
    worker = threading.Thread(target=lambda: manager._submit_background(lambda: writes.append('accepted')))
    worker.start()
    try:
        assert entered.wait(2); fence.close_admission()
        assert fence.active_count() == 1
    finally:
        release.set(); worker.join(3)
    assert writes == ['accepted']; assert fence.active_count() == 0
    manager._submit_background(lambda: writes.append('late'))
    assert writes == ['accepted']


def test_start_failure_releases_only_acquired_job(fence, monkeypatch):
    import agent.memory_lifecycle as lifecycle
    from agent.memory_lifecycle import ProviderJobs
    class Owner:
        def maintenance_drain(self, deadline): return True
    owner = Owner(); jobs = ProviderJobs(owner)
    def fail(*args, **kwargs): raise RuntimeError('synthetic construction failure')
    monkeypatch.setattr(lifecycle, 'spawn_context_thread', fail)
    with pytest.raises(RuntimeError): jobs.start(lambda: None, name='synthetic')
    assert fence.active_count() == 0


def test_unknown_result_and_inherited_capability_refuse(fence):
    from agent.memory_lifecycle import retain
    class Unknown:
        def maintenance_drain(self, deadline): return None
    retain(Unknown()); fence.close_admission()
    with pytest.raises(RuntimeError, match='Unknown'): fence.drain_owned(1)
    fresh = br.RetirementFence()
    class Child(Unknown): pass
    # The concrete child's inherited hook is not adoption.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(br, 'retirement', fresh); retain(Child())
        fresh.close_admission()
        with pytest.raises(RuntimeError, match='Unsupported'): fresh.drain_owned(1)


def test_hosted_provider_refuses_without_calling_remote_hook(fence):
    from agent.memory_lifecycle import retain
    from agent.memory_provider import MemoryProvider
    from hermes_cli.plugin_host import PluginHost
    host = object.__new__(PluginHost)
    host._generation = 1; host._release = Mock()
    host.obj_call = Mock(side_effect=AssertionError('No host RPC during proof'))
    proxy = host._object_proxy({'__object__': 1, 'methods': {'maintenance_drain': {}}}, MemoryProvider)
    retain(proxy); fence.close_admission()
    with pytest.raises(RuntimeError, match='Unsupported'): fence.drain_owned(1)
    host.obj_call.assert_not_called()


def test_real_discovery_failed_init_and_agent_removal_remain_unsupported(fence, tmp_path, monkeypatch):
    from agent.agent_init import _init_memory
    from plugins import memory
    custom = tmp_path / 'plugins' / 'synthetic_custom'; custom.mkdir(parents=True)
    (custom / '__init__.py').write_text('''from agent.memory_provider import MemoryProvider
class Provider(MemoryProvider):
    name = "retaindb"
    def is_available(self): return True
    def get_tool_schemas(self): return []
    def initialize(self, session_id, **kwargs): raise RuntimeError("synthetic init failure")
    def maintenance_drain(self, deadline): return True
''')
    monkeypatch.setattr(memory, '_external_source_dirs', lambda: [custom.parent])
    agent = SimpleNamespace(enabled_toolsets=[], disabled_toolsets=[], tools=[],
        session_id='synthetic', _session_db=None, session_cwd=None,
        _emit_startup_warning=Mock(), _emit_warning=Mock(), _emit_status=Mock())
    import agent.agent_init as init
    for name in init._GATEWAY_IDENTITY_PARAMS: setattr(agent, '_'+name, '')
    _init_memory(agent, {'memory': {'provider': 'synthetic_custom', 'memory_enabled': False,
                                  'user_profile_enabled': False}}, False, 'test')
    assert agent._memory_manager is not None
    agent._memory_manager = None
    fence.close_admission(); fence.wait_quiescent(0)
    with pytest.raises(RuntimeError, match='Unsupported'): fence.drain_owned(2)


def test_two_profile_retaindb_seed_and_prefetch_jobs_keep_owner_context(fence, tmp_path, monkeypatch):
    from plugins.memory import retaindb as rdb
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override, get_hermes_home
    calls = []; seen = set(); lock = threading.Lock()
    entered = threading.Barrier(9); release = threading.Event(); providers = []
    def http(client, method, path, **kwargs):
        identity = (str(get_hermes_home()), threading.current_thread().name)
        with lock:
            calls.append(identity[0]); first = identity not in seen; seen.add(identity)
        if first: entered.wait(3)
        assert release.wait(5)
        return SimpleNamespace(ok=True, json=lambda: {}, text='')
    monkeypatch.setattr(rdb._Client, '_http', http)
    monkeypatch.setattr(rdb, 'get_secret', lambda *a: 'synthetic')
    try:
        for profile in ('alpha', 'beta'):
            home = tmp_path / profile; home.mkdir(); (home / 'SOUL.md').write_text('synthetic identity')
            token = set_hermes_home_override(home)
            try:
                manager = MemoryManager(); provider = rdb.RetainDBMemoryProvider(); providers.append(provider)
                manager.add_provider(provider); manager.initialize_all(profile)
                manager.queue_prefetch_all('synthetic question for context')
                assert manager.flush_pending(2)
            finally: reset_hermes_home_override(token)
        entered.wait(3); fence.close_admission()
        with pytest.raises(TimeoutError): fence.wait_quiescent(0)
        assert calls.count(str(tmp_path / 'alpha')) == 4
        assert calls.count(str(tmp_path / 'beta')) == 4
    finally:
        release.set()
        for provider in providers:
            for thread in [getattr(provider, '_seed_thread', None)] + provider._prefetch_threads:
                if thread is not None: thread.join(3)
    fence.wait_quiescent(3); fence.drain_owned(3)
    assert all(provider._queue is None for provider in providers)
