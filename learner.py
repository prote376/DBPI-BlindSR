class Learner:
    # Default hyper-parameters
    update_l_rate_freq = 750
    update_l_rate_rate = 10.

    def __init__(self):
        pass

    def update(self, iteration, gan):
        if iteration == 0:
            return
        # Update learning rate every update_l_rate freq
        if iteration % self.update_l_rate_freq == 0:
            for params in gan.optimizer_G.param_groups:
                params['lr'] /= self.update_l_rate_rate
            for params in gan.optimizer_U.param_groups:
                params['lr'] /= self.update_l_rate_rate
