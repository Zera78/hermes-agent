"""Independent offline contracts for leases, retention, and uncertainty boundaries."""
import asyncio
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.tools import test_pending_completion_retention as fixtures
from tools import async_delegation as ad


class IndependentRetentionReview(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.PendingRetentionTests()
        self.fx.setUp()

    def tearDown(self):
        self.fx.tearDown()

    def test_expired_lease_reclaims_but_old_token_cannot_release_ack_or_drop(self):
        self.fx.seed()
        self.assertTrue(ad.claim_completion_delivery('synthetic-0', 'old'))
        before = self.fx.rows()
        self.assertFalse(ad.claim_completion_delivery('synthetic-0', 'new'))
        self.assertEqual(self.fx.rows(), before)
        with patch.object(ad.time, 'time', return_value=2_000_000_301.0):
            self.assertTrue(ad.claim_completion_delivery('synthetic-0', 'new'))
            before = self.fx.rows()
            for method in (ad.release_completion_delivery, ad.complete_completion_delivery, ad.drop_completion_delivery, ad.defer_completion_delivery):
                self.assertFalse(method('synthetic-0', 'old'))
                self.assertEqual(self.fx.rows(), before)
            self.assertTrue(ad.complete_completion_delivery('synthetic-0', 'new'))

    def test_eighth_live_claim_can_ack_even_though_new_claims_are_capped(self):
        self.fx.seed(spent=True, claim='eighth')
        self.assertFalse(ad.claim_completion_delivery('synthetic-0', 'ninth'))
        self.assertTrue(ad.complete_completion_delivery('synthetic-0', 'eighth'))
        self.assertEqual(self.fx.row()['delivery_state'], 'delivered')

    def test_expired_eighth_claim_is_retained_without_ninth_attempt(self):
        self.fx.seed(spent=True, claim='eighth')
        before = self.fx.rows()
        self.assertFalse(ad.claim_completion_delivery('synthetic-0', 'ninth'))
        for operation in (ad.restore_undelivered_completions, ad.sweep_orphaned_completions):
            q = queue.Queue()
            self.assertEqual(operation(q), 0)
            self.assertTrue(q.empty())
        self.assertEqual(self.fx.rows(), before)

    def test_restore_live_claim_is_only_an_offer_and_cannot_steal_delivery(self):
        self.fx.seed()
        self.assertTrue(ad.claim_completion_delivery('synthetic-0', 'holder'))
        before = self.fx.rows()
        q = queue.Queue()
        self.assertEqual(ad.restore_undelivered_completions(q), 1)
        self.assertIsNone(ad.claim_event_delivery(q.get_nowait(), 'foreign'))
        self.assertEqual(self.fx.rows(), before)
        self.assertTrue(ad.complete_completion_delivery('synthetic-0', 'holder'))

    def test_aged_delivered_requeue_preserves_payload_without_offer(self):
        self.fx.seed(aged=True, delivery='delivered')
        before = self.fx.row()
        q = queue.Queue()
        self.assertFalse(ad.requeue_admitted_completion('synthetic-0', q))
        self.assertTrue(q.empty())
        self.assertEqual(self.fx.row(), dict(before, delivery_state='pending', delivered_at=None, updated_at=ad.time.time()))

    def test_retry_verdict_is_required_to_reopen_delivered_row(self):
        from gateway.run_notifications_ownership import GatewayNotificationOwnershipMixin
        self.fx.seed(delivery='delivered')
        did = 'synthetic-0'
        key = ('async_delegation', did, '')
        runner = SimpleNamespace(_completion_delivery_lock=threading.Lock(), _completion_deliveries_delivered={key: 1})
        from tools.process_registry import process_registry
        q = queue.Queue()
        before = self.fx.rows()
        with patch.object(process_registry, 'completion_queue', q):
            asyncio.run(GatewayNotificationOwnershipMixin._requeue_uncertain_completion(runner, 'terminal', [did]))
            self.assertEqual(self.fx.rows(), before)
            self.assertIn(key, runner._completion_deliveries_delivered)
            self.assertTrue(q.empty())
            asyncio.run(GatewayNotificationOwnershipMixin._requeue_uncertain_completion(runner, 'retry', [did]))
        self.assertEqual(self.fx.row()['delivery_state'], 'pending')
        self.assertEqual(q.get_nowait()['delegation_id'], did)
        self.assertNotIn(key, runner._completion_deliveries_delivered)

    def test_retry_verdict_does_not_reoffer_spent_result(self):
        from gateway.run_notifications_ownership import GatewayNotificationOwnershipMixin
        from tools.process_registry import process_registry
        self.fx.seed(spent=True, delivery='delivered')
        runner = SimpleNamespace(_completion_delivery_lock=threading.Lock(), _completion_deliveries_delivered={})
        q = queue.Queue()
        with patch.object(process_registry, 'completion_queue', q):
            asyncio.run(GatewayNotificationOwnershipMixin._requeue_uncertain_completion(runner, 'retry', ['synthetic-0']))
        self.assertEqual(self.fx.row()['delivery_state'], 'pending')
        self.assertTrue(q.empty())

    def test_mixed_pruning_caps_settled_history_and_preserves_all_pending(self):
        for i in range(51):
            self.fx.seed(f'pending-{i}', aged=True, spent=True, claim='keep')
        pending = self.fx.rows()
        for i in range(30):
            self.fx.seed(f'delivered-{i}', delivery='delivered')
            self.fx.seed(f'dropped-{i}', delivery='dropped')
        self.fx.seed('very-old', delivery='delivered')
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute("UPDATE async_delegations SET updated_at=? WHERE delegation_id='very-old'", (ad.time.time()-ad._DURABLE_RETENTION_SECONDS-1,))
        ad._prune_durable_records()
        after = self.fx.rows()
        self.assertEqual([r for r in after if r[1].startswith('pending-')], pending)
        self.assertEqual(len(after)-len(pending), ad._MAX_RETAINED_COMPLETED)
        self.assertFalse(any(r[1] == 'very-old' for r in after))

    def test_repeated_startup_and_sweep_leave_aged_payloads_and_claims_identical(self):
        self.fx.seed(aged=True, claim='preserve-expired')
        before = self.fx.rows()
        for _ in range(3):
            for method in (ad.restore_undelivered_completions, ad.sweep_orphaned_completions):
                q = queue.Queue()
                self.assertEqual(method(q), 0)
                self.assertTrue(q.empty())
                self.assertEqual(self.fx.rows(), before)
