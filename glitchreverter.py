import copy


class GlitchReverter:
    def __init__(self, model):
        self.model = model
        self.checkpoint = None

    def save_checkpoint(self):
        self.checkpoint = copy.deepcopy(self.model.Wout)

    def revert_if_glitch(self, glitch_active, loss_spike):
        if glitch_active and loss_spike and self.checkpoint is not None:
            self.model.Wout = copy.deepcopy(self.checkpoint)
            return True
        return False
