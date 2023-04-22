from typing import *


class RLAgent:
    def __init__(self, decision_history: list[dict] = []):
        self.memory = decision_history

    def update_action_state(self, action_state: dict):
        self.memory.append(action_state)

    def take_action(self):
        # todo: need to implement environment first
        pass

    def learn(self):
        # todo: need to implement environment first
        pass
