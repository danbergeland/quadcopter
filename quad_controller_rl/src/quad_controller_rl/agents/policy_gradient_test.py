import unittest
import numpy as np
from policy_gradient import DDPG
from gym import spaces

class ddpg_tests(unittest.TestCase):

    def setUp(self):
        self.gymtask = task_mock()
        self.agent = DDPG(self.gymtask)

    def test_DDPG_init_params(self):
        self.assertEqual(self.agent.action_size, (2,))
        self.assertEqual(self.agent.state_size, (3,))

    def test_step_returns_action(self):
        state = np.array([0,0,0])
        action = self.agent.step(state, -5.0, False)
        self.assertIsNotNone(action)

    def test_step_updates_lastState_vars(self):
        state = np.array([0,0,0])
        action = self.agent.step(state, -5.0, False)
        self.assertIsNotNone(action)
        self.assertTrue(np.array_equal(self.agent.last_action, action))
        self.assertTrue(np.array_equal(self.agent.last_state, state))

    def test_reset_clears_state_vars(self):
        state = np.array([0,0,0])
        action = self.agent.step(state, -5.0, False)
        action = self.agent.step(state, -4.0, True)
        self.assertEqual(self.agent.last_state, None)
        self.assertEqual(self.agent.last_action, None)
        
        



#Helper mock for task passed to agent at init 
class task_mock(object):
    def __init__(self):
        self.observation_space = spaces.Box(low=np.array([-10.0,-10.0,-10.0]), high=np.array([10.0,10.0,10.0]))
        self.action_space = spaces.Box(low=np.array([-10.0,-10.0]), high=np.array([10.0,10.0]))

if __name__ == '__main__':
    unittest.main()



