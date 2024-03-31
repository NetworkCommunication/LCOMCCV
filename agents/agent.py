class Agent(object):

    NAME = "Abstract Agent"

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, state):
        raise NotImplementedError

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        raise NotImplementedError

    def __str__(self):
        desc = self.NAME
        return desc
