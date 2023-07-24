class WeightedCombinationOfLosses:
    def __init__(self, list_loss_funcs, list_weights=None):
        self.list_loss_funcs = list_loss_funcs
        if list_weights is None:
            self.list_weights = [1.0 for _ in range(len(list_loss_funcs))]
        else:
            self.list_weights = list_weights

    def __call__(self, im1, im2):
        loss = 0.0
        weight_sum = 0
        for i, lf in enumerate(self.list_loss_funcs):
            loss += self.list_weights[i] * lf(im1, im2)
            weight_sum += self.list_weights[i]

        return loss / weight_sum
