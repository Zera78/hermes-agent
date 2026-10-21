"""Synthetic offline regression: automatic retry limits must not discard durable results."""
import json
import queue
import threading
import time
import unittest
from unittest.mock import patch

from hermes_constants import get_hermes_home
from tools import async_delegation as ad


class PendingRetentionTests(unittest.TestCase):
    def setUp(self):
        ad._reset_for_tests()
        self.clock = patch.object(ad.time, 'time', return_value=2_000_000_000.0)
        self.clock.start()
        self.liveness = patch.object(ad, '_owner_liveness', return_value=lambda pid, started: pid == 123)
        self.liveness.start()
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute('DELETE FROM async_delegations')

    def tearDown(self):
        self.liveness.stop()
        self.clock.stop()
        ad._reset_for_tests()

    def seed(self, did='synthetic-0', *, aged=False, spent=False, delivery='pending', claim=None):
        completed = ad.time.time() - (ad._MAX_COMPLETION_REPLAY_AGE_S + 1 if aged else 100)
        event = {'type': 'async_delegation', 'delegation_id': did, 'session_key': 'synthetic-owner',
                 'origin_ui_session_id': 'synthetic-ui', 'status': 'completed', 'summary': 'SYNTHETIC ONLY',
                 'dispatched_at': completed - 10, 'completed_at': completed}
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute('''INSERT INTO async_delegations
                (delegation_id, origin_session, origin_ui_session_id, parent_session_id, state,
                 dispatched_at, completed_at, updated_at, event_json, result_json, delivery_state,
                 delivery_attempts, delivered_at, owner_pid, owner_started_at, task_json,
                 delivery_claim, delivery_claimed_at, origin_session_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (did, 'synthetic-owner', 'synthetic-ui', 'synthetic-parent', 'completed',
                 completed - 10, completed, completed, json.dumps(event), json.dumps({'summary': 'SYNTHETIC ONLY'}),
                 delivery, ad._MAX_DELIVERY_ATTEMPTS if spent else 1,
                 completed if delivery == 'delivered' else None, 0, 7, '{"goal":"synthetic"}',
                 claim, completed if claim else None, 'synthetic-origin'))
        return event

    def rows(self):
        with ad._DB_LOCK, ad._transaction() as conn:
            return conn.execute('SELECT rowid,* FROM async_delegations ORDER BY delegation_id').fetchall()

    def row(self, did='synthetic-0'):
        with ad._DB_LOCK, ad._transaction() as conn:
            cur = conn.execute('SELECT rowid,* FROM async_delegations WHERE delegation_id=?', (did,))
            return dict(zip([d[0] for d in cur.description], cur.fetchone()))

    def assert_parked(self, method, *, aged=False, spent=False):
        for i in range(5):
            self.seed(f'synthetic-{i}', aged=aged, spent=spent)
        before = self.rows()
        q = queue.Queue()
        self.assertEqual(method(q), 0)
        self.assertTrue(q.empty())
        self.assertEqual(self.rows(), before, 'all typed row values, rowids, payload bytes and claims must survive')

    def test_aged_startup_preserves_five_pending_results_without_offering(self):
        self.assert_parked(ad.restore_undelivered_completions, aged=True)

    def test_aged_sweep_preserves_five_pending_results_without_offering(self):
        self.assert_parked(ad.sweep_orphaned_completions, aged=True)

    def test_spent_startup_preserves_five_pending_results_without_offering(self):
        self.assert_parked(ad.restore_undelivered_completions, spent=True)

    def test_spent_sweep_preserves_five_pending_results_without_offering(self):
        self.assert_parked(ad.sweep_orphaned_completions, spent=True)

    def test_recent_result_remains_offerable_and_requires_an_owner(self):
        from tools.process_registry import ProcessRegistry
        event = self.seed()
        before = self.rows()
        q = queue.Queue()
        self.assertEqual(ad.restore_undelivered_completions(q), 1)
        offered = q.get_nowait()
        self.assertEqual(offered, dict(event, restored=True, startup_replay=True))
        self.assertEqual(self.rows(), before)
        self.assertFalse(ProcessRegistry._owns_event(offered, '', None, True))
        self.assertTrue(ProcessRegistry._owns_event(offered, 'synthetic-owner', None, True))
        self.assertEqual(ad.sweep_orphaned_completions(q), 0)

    def test_fresh_owner_and_live_claim_are_not_swept(self):
        self.seed('claimed', claim='synthetic-claim')
        self.seed('owned')
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute("UPDATE async_delegations SET owner_pid=123 WHERE delegation_id='owned'")
            conn.execute("UPDATE async_delegations SET delivery_claimed_at=? WHERE delegation_id='claimed'", (ad.time.time(),))
        before = self.rows()
        q = queue.Queue()
        self.assertEqual(ad.sweep_orphaned_completions(q), 0)
        self.assertEqual(self.rows(), before)
        self.assertTrue(q.empty())

    def test_failed_final_attempt_releases_claim_and_keeps_result_pending(self):
        self.seed(spent=True, claim='synthetic-claim')
        before = self.row()
        self.assertTrue(ad.release_completion_delivery('synthetic-0', 'synthetic-claim'))
        expected = dict(before, delivery_claim=None, delivery_claimed_at=None, updated_at=ad.time.time())
        self.assertEqual(self.row(), expected)

    def test_released_final_attempt_cannot_be_claimed_again(self):
        self.seed(spent=True, claim='synthetic-claim')
        self.assertTrue(ad.release_completion_delivery('synthetic-0', 'synthetic-claim'))
        before = self.rows()
        self.assertFalse(ad.claim_completion_delivery('synthetic-0', 'synthetic-ninth'))
        self.assertEqual(self.rows(), before)

    def test_real_completion_persistence_leaves_delivery_pending(self):
        event = {'type': 'async_delegation', 'delegation_id': 'real-producer-synthetic-data',
                 'session_key': 'synthetic-owner', 'status': 'completed', 'summary': 'SYNTHETIC ONLY',
                 'dispatched_at': ad.time.time() - 10, 'completed_at': ad.time.time()}
        with patch('gateway.status.get_process_start_time', return_value=7):
            ad._persist_dispatch(event)
        ad._persist_completion(event, {'status': 'completed', 'summary': 'SYNTHETIC ONLY'})
        row = self.row('real-producer-synthetic-data')
        self.assertEqual((row['state'], row['delivery_state']), ('completed', 'pending'))
        self.assertIsNone(row['delivered_at'])
        self.assertIsNone(row['delivery_claim'])

    def test_wrong_release_token_cannot_mutate_pending_result(self):
        self.seed(spent=True, claim='synthetic-claim')
        before = self.rows()
        self.assertFalse(ad.release_completion_delivery('synthetic-0', 'wrong'))
        self.assertEqual(self.rows(), before)

    def test_uncertain_spent_admission_returns_to_pending_without_offering(self):
        self.seed(spent=True, delivery='delivered')
        before = self.row()
        q = queue.Queue()
        self.assertFalse(ad.requeue_admitted_completion('synthetic-0', q))
        expected = dict(before, delivery_state='pending', delivered_at=None, updated_at=ad.time.time())
        self.assertEqual(self.row(), expected)
        self.assertTrue(q.empty())

    def test_history_pruning_never_deletes_pending_results(self):
        for i in range(5):
            self.seed(f'synthetic-{i}', aged=True, spent=True)
        before = self.rows()
        for i in range(4):
            self.seed(f'settled-{i}', delivery='delivered')
        with patch.object(ad, '_MAX_RETAINED_COMPLETED', 2):
            ad._prune_durable_records()
        pending = [r for r in self.rows() if r[1].startswith('synthetic-')]
        self.assertEqual(pending, before)
        self.assertLessEqual(len(self.rows()) - len(pending), 2)

    def test_pending_overflow_does_not_delete_results(self):
        self.seed()
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute("""WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<1000)
                INSERT INTO async_delegations
                    (delegation_id,origin_session,state,dispatched_at,updated_at,event_json,result_json,delivery_state)
                SELECT 'synthetic-overflow-'||i,'synthetic-owner','completed',1,1,'{}','{}','pending' FROM n""")
        before = self.rows()
        with patch.object(ad, '_MAX_RETAINED_COMPLETED', 2000):
            ad._prune_durable_records()
        self.assertEqual(len(self.rows()), 1001)
        self.assertEqual(self.rows(), before)

    def test_explicit_permanent_boundary_can_still_drop_its_claim(self):
        self.seed(claim='synthetic-boundary')
        self.assertTrue(ad.drop_completion_delivery('synthetic-0', 'synthetic-boundary'))
        self.assertEqual(self.row()['delivery_state'], 'dropped')
        self.assertEqual(self.row()['result_json'], '{"summary": "SYNTHETIC ONLY"}')

    def test_recent_result_atomic_claim_and_ack_still_work(self):
        self.seed()
        self.assertTrue(ad.claim_completion_delivery('synthetic-0', 'synthetic-first'))
        self.assertFalse(ad.claim_completion_delivery('synthetic-0', 'synthetic-second'))
        self.assertTrue(ad.complete_completion_delivery('synthetic-0', 'synthetic-first'))
        self.assertEqual(self.row()['delivery_state'], 'delivered')

    def test_actual_poller_startup_with_stop_already_set_preserves_pending_rows(self):
        from tui_gateway import server
        from tools.process_registry import process_registry
        for i in range(5):
            self.seed(f'synthetic-{i}', aged=True)
        before = self.rows()
        stop = threading.Event()
        stop.set()
        q = queue.Queue()
        with patch.object(process_registry, 'completion_queue', q), patch.object(process_registry, '_completions_restored', False):
            server._notification_poller_scoped_loop(stop, 'synthetic-ui', {'_notification_emitted': set()})
        self.assertEqual(self.rows(), before)
        self.assertTrue(q.empty())


if __name__ == '__main__':
    unittest.main(verbosity=2)
