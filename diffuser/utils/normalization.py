import numpy as np

class StateNormalizer:
    def __init__(self, dataset):
        """
        dataset: A numpy array of shape [num_samples, state_dim]
        """
        self.dataset = dataset
        self.cdfs = self._compute_cdfs(dataset)

    def _compute_cdfs(self, dataset):
        """
        Compute empirical CDFs for each state dimension.
        """
        cdfs = []
        for dim in range(dataset.shape[-1]):
            sorted_states = np.sort(dataset[:, dim])
            cdf = lambda x: np.searchsorted(sorted_states, x, side='right') / len(sorted_states)
            cdfs.append(cdf)
        return cdfs

    def normalize(self, states):
        """
        Normalize states to [-1, 1] using the empirical CDF.
        states: A numpy array of shape [batch_size, state_dim]
        """
        norm_states = []
        for i, cdf in enumerate(self.cdfs):
            norm_states.append(2 * cdf(states[:, i]) - 1)
        return np.stack(norm_states, axis=-1)

    def unnormalize(self, norm_states):
        """
        Unnormalize states from [-1, 1] to the original scale.
        """
        unnorm_states = []
        for i in range(len(self.cdfs)):
            sorted_states = np.sort(self.dataset[:, i])
            inv_cdf = lambda x: sorted_states[int(((x + 1) / 2) * len(sorted_states))]
            unnorm_states.append([inv_cdf(xi) for xi in norm_states[:, i]])
        return np.stack(unnorm_states, axis=-1)

