import unittest
from simulator.scenarios import get_scenario_suite
from simulator.environment import RFEnvironment, Emitter
from simulator.multi_receiver import SpatialRFEnvironment, MultiReceiverSystem


class TestScenarioSuite(unittest.TestCase):
    def setUp(self):
        self.scenarios = get_scenario_suite()

    def test_all_ten_scenarios_present(self):
        """Must contain all required scenarios S1 through S10."""
        expected_ids = [f"S{i}" for i in range(1, 11)]
        actual_ids = [s["id"] for s in self.scenarios.values()]
        for eid in expected_ids:
            self.assertIn(eid, actual_ids)

    def test_scenarios_execute_in_environment(self):
        """Each scenario must cleanly initialize and step in simulation environment."""
        for name, spec in self.scenarios.items():
            if len(spec["receivers"]) == 1:
                env = RFEnvironment(num_bands=10, seed=42)
                for e_spec in spec["emitters"]:
                    kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
                    e = Emitter(name=e_spec['name'], band=e_spec['band'], behavior=e_spec['behavior'], seed=42, **kwargs)
                    env.add_emitter(e)
                for _ in range(5):
                    env.step()
                self.assertEqual(env.time, 5)
            else:
                env = SpatialRFEnvironment(num_bands=10, seed=42)
                for e_spec in spec["emitters"]:
                    kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
                    e = Emitter(name=e_spec['name'], band=e_spec['band'], behavior=e_spec['behavior'], seed=42, **kwargs)
                    env.add_emitter(e, position=e_spec.get('position', (0, 0)))
                mrs = MultiReceiverSystem(env, spec["receivers"])
                for _ in range(5):
                    mrs.scan(spec["receivers"][0]["id"], 0, dwell_time=1)
                self.assertGreaterEqual(env.time, 5)


if __name__ == '__main__':
    unittest.main()
