"""Process-owned manager/provider lifetimes for strict maintenance shutdown.

Ordinary shutdown remains best effort. These records never call session extraction
or discover providers: only objects encountered during admitted work participate.
"""
from contextlib import contextmanager
from functools import wraps
import threading
import time

from agent.memory_provider import spawn_context_thread


class MemoryAdmissionClosed(RuntimeError):
    pass


def fence():
    from hermes_cli.backend_retirement import retirement
    return retirement


def retain(owner):
    """Capability must be declared on the concrete class, never inherited by proxies.

    Returning None, a name, or an ordinary shutdown method is not a capability.
    Third-party adopters require separate integration review before claiming support.
    """
    from agent.memory_provider import MemoryProvider
    if isinstance(owner, MemoryProvider):
        return retain_provider(owner)
    concrete = type(owner).__dict__.get('maintenance_drain')
    hosted = '__hermes_plugin_host_ref__' in type(owner).__dict__
    proof = (lambda deadline: concrete(owner, deadline)) if callable(concrete) and not hosted else None
    fence().retain_drain(owner, proof)


def retain_provider(provider):
    """Only the reviewed integration boundary may supply a provider's proof.

    Keep unsupported records before calls and across manager reuse, even when the
    provider declares a positive hook. Internal resource hooks remain separate.
    """
    from plugins.memory import reviewed_provider_drain
    concrete = reviewed_provider_drain(provider)
    hosted = '__hermes_plugin_host_ref__' in type(provider).__dict__
    proof = (lambda deadline: concrete(provider, deadline)) if callable(concrete) and not hosted else None
    fence().retain_drain(provider, proof)


@contextmanager
def memory_work():
    reservation = fence().reserve()
    if reservation is None:
        raise MemoryAdmissionClosed('Memory maintenance admission is closed')
    try:
        with reservation.scope(): yield reservation
    finally:
        reservation.finish()


def owned_call(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        with memory_work(): return fn(*args, **kwargs)
    return call


class ProviderJobs:
    """Retains every finite provider job, including superseded public handles."""
    def __init__(self, owner):
        self._lock = threading.Lock()
        self._threads = []
        self._fence = fence()
        retain_provider(owner)

    def start(self, target, *, name, args=()):
        reservation = self._fence.reserve()
        if reservation is None:
            raise MemoryAdmissionClosed('Provider job admission is closed')
        def run():
            try:
                with reservation.scope(): target(*args)
            finally:
                reservation.finish()
        thread = None
        try:
            thread = spawn_context_thread(run, name=name)
            with self._lock: self._threads.append(thread)
            thread.start()
        except Exception:
            # A definite Thread.start failure has no executing target. Interrupted
            # starts (BaseException) remain unknown until the actual target settles.
            if thread is None or thread.ident is None:
                reservation.finish()
                with self._lock:
                    if thread is not None: self._threads.remove(thread)
            raise
        return thread

    def drain(self, deadline):
        with self._lock: threads = tuple(self._threads)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive(): raise TimeoutError('Provider jobs still active')
        return True


class MemoryStartup:
    """Discovery/import failure cannot erase an encountered integration."""
    def __init__(self):
        self.resolved = False
        retain(self)

    def maintenance_drain(self, deadline):
        if not self.resolved:
            raise RuntimeError('Unsupported memory lifecycle: unresolved provider initialization')
        return True


def adopt_manager(manager):
    # Exact participating instances can be reused without initialization or extraction.
    # A custom/injected object or a foreign lifetime cannot furnish this proof.
    from agent.memory_manager import MemoryManager
    if type(manager) is MemoryManager and manager._maintenance_fence is fence():
        retain(manager)
        # Reuse is an actual provider-adoption boundary, including providers
        # installed by an injected list rather than public add_provider().
        for provider in manager.providers:
            retain_provider(provider)
    else:
        fence().retain_drain(manager, None)
