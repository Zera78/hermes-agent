"""Process ownership around automatic, deferred and explicit review entry points."""
from __future__ import annotations
from typing import Any, Optional

def spawn_review(self, messages_snapshot: list[dict], review_memory: bool = False,
                             review_skills: bool = False, focus: Optional[str] = None, explicit: bool = False) -> None:
    """Post-turn review entry point: decide WHEN, then spawn.

    A review whose runtime is the MANAGED LOCAL llama-server is queued for machine idle (``defer: auto``)
    instead of hitting the user's GPU mid-session; everything else spawns immediately. ``explicit``
    (/refine) is never deferred but does not touch the ``focus``-keyed delegate/enabled gates.
    """
    from hermes_cli.backend_retirement import retirement
    with retirement.work() as admitted:
        if admitted:
            _schedule_review(self, messages_snapshot, review_memory, review_skills, focus, explicit)


def _schedule_review(self, messages_snapshot, review_memory, review_skills, focus, explicit):
    from run_agent import _review_should_defer, _review_queue_key
    # Gates run at enqueue/spawn time; the idle dispatcher re-checks `enabled` at dispatch time.
    if focus is None and getattr(self, "_delegate_depth", 0) > 0:
        return
    task_cfg = None
    if focus is None:
        from agent.background_review import load_background_review_settings
        enabled, task_cfg = load_background_review_settings()
        if not enabled:
            return

    # Structural clone at the single chokepoint: the fork sanitizes in place, and a shallow copy would
    # alias the live history's nested tool_calls/content.
    # Structural clone at the single chokepoint every review path (automatic, /refine, idle-queue
    # deferral) goes through. See #100795.
    from agent.turn_finalizer import _clone_background_review_messages
    kwargs = dict(messages_snapshot=_clone_background_review_messages(messages_snapshot),
                  review_memory=review_memory, review_skills=review_skills, focus=focus, task_cfg=task_cfg,
                  explicit=explicit)
    if focus is None and not explicit and _review_should_defer(self, task_cfg):
        from agent.review_idle_queue import QUEUE
        QUEUE.enqueue(self, _review_queue_key(self), kwargs)
        return
    self._spawn_background_review_now(**kwargs)

def spawn_review_now(self, messages_snapshot: list[dict], review_memory: bool = False,
                                 review_skills: bool = False, focus: Optional[str] = None,
                                 task_cfg: Optional[dict[str, Any]] = None, _requeue_attempts: int = 0,
                                 explicit: bool = False, _wait_for_slot: bool = False) -> None:
    """Spawn the background memory/skill review thread.

    ``threading.Thread`` is constructed here so tests patching ``run_agent.threading.Thread`` keep working.
    ``focus`` is /refine steering text; ``task_cfg`` is the pre-loaded config block (None on direct calls).
    ``explicit`` (/refine) forks under the ``refine_review`` write origin, keeping the full
    memory operation set. A deferred review preempted by a live turn is requeued (bounded)
    rather than lost.
    """
    from agent.background_review import (
        finish_background_review_run, prepare_background_review_run, spawn_background_review_thread,
    )
    from tools.thread_context import propagate_context_to_thread

    from run_agent import threading
    from hermes_cli.backend_retirement import retirement
    ownership = retirement.reserve()
    if ownership is None:
        return
    review_run = None
    try:
        review_run = prepare_background_review_run(self, wait_for_slot=_wait_for_slot)
        if review_run is None:
            ownership.finish()
            return
        target, _prompt = spawn_background_review_thread(
            self, messages_snapshot, review_memory=review_memory, review_skills=review_skills,
            focus=focus, task_cfg=task_cfg, review_run=review_run, explicit=explicit,
        )

        def _target_with_requeue() -> None:
            try:
                with ownership.scope():
                    try:
                        target()
                    finally:
                        # Request acknowledgement and complete writer settlement are distinct.
                        finish_background_review_run(self, review_run)
                    self._maybe_requeue_preempted_review(review_run, dict(
                        messages_snapshot=messages_snapshot, review_memory=review_memory, review_skills=review_skills,
                        focus=focus, task_cfg=task_cfg, _requeue_attempts=_requeue_attempts + 1,
                        explicit=explicit))
            finally:
                ownership.finish()

        # Carry the active profile into the review thread so MEMORY.md / skill review writes land in the
        # right profile.
        threading.Thread(target=propagate_context_to_thread(_target_with_requeue), daemon=True, name="bg-review").start()
    except Exception:
        # Thread.start interruption can leave its outcome unknown; only definite
        # startup failures unwind here. A running target settles its own reservation.
        finish_background_review_run(self, review_run)
        ownership.finish()
        raise
