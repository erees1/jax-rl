class ActorCritic(BaseAgent):
    def __init__(self, layer_spec, **kwargs):
        super.__init__(layer_spec, **kwargs)

    def update(self, batch_size):
        self.steps_trained +=1 
        (observations, actions, rewards, next_observations, dones) = self.buffer.sample_batch(
            batch_size
        )
