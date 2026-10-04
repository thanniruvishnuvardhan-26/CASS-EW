"""
Unit tests for RF Knowledge Map, Uncertainty, Anti-Starvation, and Explainability.
"""

import unittest
import numpy as np

from algorithms.rf_knowledge_map import RFKnowledgeMap, BandKnowledgeEntry
from algorithms.phase8_spatial_scheduler import SpatialScheduler, ActionExplanation


class TestRFKnowledgeMap(unittest.TestCase):

    def setUp(self):
        self.receiver_ids = ['R1', 'R2', 'R3']
        self.num_bands = 10
        self.km = RFKnowledgeMap(self.receiver_ids, self.num_bands, max_revisit_threshold=30)

    def test_initialization(self):
        """Test proper default state across all receivers and bands."""
        self.assertEqual(len(self.km.receiver_ids), 3)
        self.assertEqual(self.km.num_bands, 10)
        entry = self.km.get_entry('R1', 0)
        self.assertEqual(entry.belief, 0.5)
        self.assertEqual(entry.uncertainty, 1.0)
        self.assertEqual(entry.staleness, 0)
        self.assertEqual(entry.starvation_count, 0)
        self.assertEqual(entry.activity_state, "UNKNOWN")

    def test_record_scan_and_staleness(self):
        """Test scan recording, state updates, and staleness accumulation."""
        # Record detection at t=5 on (R1, band 2)
        self.km.record_scan_result(
            receiver_id='R1',
            band=2,
            obs_time=5,
            detected=True,
            belief=0.8,
            temporal_score=0.9,
            prediction_score=0.85,
            prediction_confidence=0.7,
            spatial_evidence=0.6
        )

        entry = self.km.get_entry('R1', 2)
        self.assertEqual(entry.last_observation, 5)
        self.assertEqual(entry.last_detection, 5)
        self.assertEqual(entry.activity_state, "ACTIVE")
        self.assertEqual(entry.starvation_count, 0)
        self.assertLess(entry.uncertainty, 0.8)  # Uncertainty reduced after high-confidence detection

        # Advance time to t=25
        self.km.update_staleness(25)
        self.assertEqual(self.km.get_entry('R1', 2).staleness, 20)
        # Unvisited bands have staleness 25
        self.assertEqual(self.km.get_entry('R2', 0).staleness, 25)

    def test_principled_uncertainty_growth(self):
        """Uncertainty should increase with staleness and decrease with high confidence."""
        entry = self.km.get_entry('R1', 3)
        entry.belief = 0.95
        entry.staleness = 0
        entry.prediction_confidence = 0.9
        unc_fresh = self.km.compute_uncertainty(entry)

        entry.staleness = 100
        unc_stale = self.km.compute_uncertainty(entry)
        self.assertGreater(unc_stale, unc_fresh)

    def test_starvation_tracking(self):
        """Bands not visited must accumulate starvation and be flagged."""
        # Scan R1 band 0 repeatedly
        for t in range(1, 40):
            self.km.record_scan_result('R1', 0, obs_time=t, detected=False, belief=0.4)

        starving = self.km.get_starving_bands(threshold=30)
        # All other bands should be in starving list since they have staleness >= 30
        self.assertIn(('R2', 1), starving)
        self.assertIn(('R3', 5), starving)
        self.assertNotIn(('R1', 0), starving)


class TestSchedulerExplainabilityAndAntiStarvation(unittest.TestCase):

    def setUp(self):
        self.receiver_ids = ['R1', 'R2']
        self.num_bands = 5

    def test_decision_explanation_structure(self):
        """Verify scheduler records mathematically verified explanations for decisions."""
        sched = SpatialScheduler(
            receiver_ids=self.receiver_ids,
            num_bands=self.num_bands,
            spatial_weight=0.3,
            seed=42,
            epsilon=0.0
        )
        # Provide an observation
        obs = {'time': 1, 'observation_time': 1}
        action = sched.select_action(obs)

        expl = sched.explain_last_decision()
        self.assertIn('receiver', expl)
        self.assertIn('band', expl)
        self.assertIn('total_priority', expl)
        self.assertIn('belief_contribution', expl)
        self.assertIn('reason', expl)
        self.assertEqual(expl['receiver'], action[0])
        self.assertEqual(expl['band'], action[1])

    def test_anti_starvation_guarantee(self):
        """Verify scheduler forces visit to starving band when max_revisit_interval is exceeded."""
        sched = SpatialScheduler(
            receiver_ids=['R1'],
            num_bands=3,
            spatial_weight=0.0,
            seed=42,
            epsilon=0.0,
            max_revisit_interval=10
        )

        # Continually reinforce band 0 so it has maximum belief
        for t in range(1, 15):
            sched.update({'time': t, 'observation_time': t}, ('R1', 0), result=True)

        # At t=15, bands 1 and 2 have been unvisited for 15 steps > 10
        # The scheduler must select a starving band (band 1 or 2), not band 0
        action = sched.select_action({'time': 15, 'observation_time': 15})
        self.assertIn(action[1], [1, 2])
        self.assertTrue(sched.last_explanation.is_anti_starvation)


if __name__ == '__main__':
    unittest.main()
