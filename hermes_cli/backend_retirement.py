"""Process-owned admission fence for cooperative Desktop backend retirement.

The gate belongs to the process, not a profile or transport. Diagnostic ledgers alone cannot
close the check/use race: every source of new work must reserve admission before dispatch.
"""

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import logging
import secrets
import threading
import time


_current_work = ContextVar("retirement_work", default=None)


class WorkReservation:
    """One process-owned lifetime; descendants reserve before this owner settles."""

    def __init__(self, fence):
        self._fence = fence
        self._finished = False

    @contextmanager
    def scope(self):
        token = _current_work.set(self)
        try:
            yield self
        finally:
            _current_work.reset(token)

    def finish(self):
        # Idempotent for a synchronous test executor or a start failure after unwind.
        with self._fence._quiet:
            if not self._finished:
                self._finished = True
                self._fence._active -= 1
                self._fence._quiet.notify_all()


class RetirementFence:
    """Exclusive 30-second prepare permit; commit closes admission permanently.

    Tokens are random and recognized only by this process instance. A committed prepare
    returns the same token so a client can recover a lost commit response without reopening.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._quiet = threading.Condition(self._lock)
        self._closed = False
        self._active = 0
        self._legacy_active = 0
        self._preparing = False
        self._token = None
        self._committed = False
        self._expires_at = 0.0
        self._now = time.monotonic
        self._drains = {}

    def retain_drain(self, owner, proof):
        """Retain a resource before its side effects, independently of sessions/handles.

        Registration is not admission: the caller must already hold a reservation.
        Unsupported classification is permanent for this owner, even when a
        generic proof was previously retained or cached as settled.
        """
        with self._lock:
            record = self._drains.get(id(owner))
            if record is None:
                self._drains[id(owner)] = [owner, proof, copy_context(), False]
            elif proof is None:
                record[1], record[3] = None, False

    def drain_owned(self, timeout):
        """Run local resource proofs after producer settlement, using one deadline.

        The host retains the off-loop caller if a proof blocks. Retrying therefore
        cannot overlap an unfinished hook. Only the singleton True proves settlement.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                records = [record for record in self._drains.values() if not record[3]]
            if not records:
                return
            for record in records:
                with self._lock:
                    owner, proof, context, _settled = record
                self._check_drain(owner, proof, context, deadline)
                # Cache only an unchanged proof. A restrictive classification can
                # arrive from another resource proof (or while this proof runs).
                # The next pass must observe it even if this owner was checked.
                with self._lock:
                    if record[1] is proof:
                        record[3] = True

    @staticmethod
    def _check_drain(owner, proof, context, deadline):
        label = type(owner).__name__
        if proof is None:
            raise RuntimeError('Unsupported memory lifecycle: ' + label)
        if time.monotonic() >= deadline:
            raise TimeoutError('Memory lifecycle deadline: ' + label)
        if context.run(proof, deadline) is not True:
            raise RuntimeError('Unknown memory lifecycle result: ' + label)

    def _expire(self):
        if self._token is not None and not self._committed and self._now() >= self._expires_at:
            self._token = None

    def acquire(self):
        with self._lock:
            self._expire()
            if self._closed or self._preparing or self._token is not None:
                return False
            self._active += 1
            self._legacy_active += 1
            return True

    def release(self):
        with self._lock:
            if self._legacy_active <= 0:
                raise RuntimeError("Unacquired retirement release")
            self._legacy_active -= 1
            self._active -= 1
            self._quiet.notify_all()

    def active_count(self):
        with self._lock:
            return self._active

    def reserve(self):
        """Admit a root, or a child of the current still-owned scope, atomically.

        A copied ContextVar is not authority once its reservation has settled. Children
        must reserve before publication; no count-zero gap can cross close/prepare.
        """
        parent = _current_work.get()
        with self._lock:
            self._expire()
            descendant = (parent is not None and parent._fence is self and not parent._finished)
            if not descendant and (self._closed or self._preparing or self._token is not None):
                return None
            self._active += 1
            return WorkReservation(self)

    @contextmanager
    def work(self):
        reservation = self.reserve()
        try:
            if reservation is None:
                yield False
            else:
                with reservation.scope():
                    yield True
        finally:
            if reservation is not None:
                reservation.finish()

    def close_admission(self):
        """Permanent process-local maintenance stop; admitted work must still unwind."""
        with self._lock:
            self._closed = True

    def is_closed(self):
        with self._lock:
            return self._closed

    def wait_quiescent(self, timeout):
        """A timeout is uncertainty, never permission to tear down shared resources."""
        with self._quiet:
            owner = _current_work.get()
            if owner is not None and owner._fence is self and not owner._finished:
                raise RuntimeError("Owned work cannot wait for its own settlement")
            if not self._closed:
                raise RuntimeError("Maintenance admission is still open")
            if not self._quiet.wait_for(lambda: self._active == 0, timeout=max(0.0, timeout)):
                raise TimeoutError("Admitted native writers did not stop")

    def run_if_admitted(self, fn):
        with self.work() as admitted:
            if admitted:
                return fn()

    def prepare(self):
        from hermes_cli.web_server_idle_proof import idle_proof

        with self._lock:
            self._expire()
            if self._committed:
                return {"ok": True, "idle": True, "token": self._token}
            if self._closed or self._preparing or self._token is not None or self._active:
                return {"ok": False, "idle": False}
            self._preparing = True
        # No admissions can cross the freeze. Do not hold our lock while reading ledgers:
        # some callers already hold a session/cron lock when reserving their work.
        verdict = None
        try:
            verdict = idle_proof()["idle"]
        except Exception:
            logging.getLogger(__name__).warning("Retirement idle probe unavailable", exc_info=True)
        finally:
            with self._lock:
                self._preparing = False
                if verdict is True:
                    self._token = token = secrets.token_urlsafe(32)
                    self._expires_at = self._now() + 30.0
        if verdict is not True:
            return {"ok": False, "idle": verdict}
        return {"ok": True, "idle": True, "token": token}

    def commit(self, token):
        with self._lock:
            self._expire()
            if not token or token != self._token:
                return {"ok": False}
            self._committed = True
            return {"ok": True}

    def cancel(self, token):
        with self._lock:
            self._expire()
            if not token or token != self._token or self._committed:
                return {"ok": False}
            self._token = None
            return {"ok": True}


retirement = RetirementFence()
