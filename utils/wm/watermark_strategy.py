from abc import ABC, abstractmethod
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.special import binom
import galois
from scipy.special import erf
from .wm_provider import WmProvider


class WatermarkStrategy(ABC):
    @abstractmethod
    def get_key(self):
        """
        return the cipher(the minimum knowledge that is shared between the encoder and decoder)
        :return: a dict of all necessary info
        """
        pass

    @abstractmethod
    def get_init_latent(self):
        """
        return the initial latent vector, the shape is consistent by the actual impl(e.g. [1, 4, 64, 64])
        :return: some tensor
        """
        pass

    @abstractmethod
    def detect(self, reversed_w):
        """
        detect the if the watermark exists in the latent
        :param reversed_w: the reversed latent, we want to detect the watermark in it
        :return: the probability of the watermark exists in the latent
        """
        pass

    @abstractmethod
    def get_tpr(self):
        """
        get the true positive rate.
        the state should be maintained via detect function
        :return:  the true positive rate
        """
        pass


class PRCWatermark(WatermarkStrategy, WmProvider):
    def __init__(self, fpr = 0.00001, prc_t = 3, letent_length = 16384, message_length = 512, var = 1.5, basis = None):
        super().__init__()
        self.fpr = fpr
        self.prc_t = prc_t
        self.latent_length = letent_length  # 4 * 64 * 64
        self.encoding_key, self.decoding_key = None, None
        self.GF = galois.GF(2)
        self.message_length = message_length
        self.var = var # for decoding
        self.basis = basis # for decoding / sampling
        self.message = None
        self.tp = 0
        self.num = 0

    def get_key(self):
        """
        return the cipher(the minimum knowledge that is shared between the encoder and decoder)
        :return: a dict of all necessary info
        """
        if self.encoding_key is None:
            self.encoding_key, self.decoding_key = self.KeyGen()
        return {
            'encoding_key': self.encoding_key,
            'decoding_key': self.decoding_key
        }


    def get_init_latent(self, dim = (1, 4, 64, 64)):
        """
        return the initial latent vector, the shape is consistent by the actual impl(e.g. [1, 4, 64, 64])
        :return: some tensor on cpu
        """
        prc_codeword = self.encode()
        return self.sample(prc_codeword).reshape(*dim)


    def detect(self, reversed_w):
        """
        detect if the watermark exists in the latent
        :param reversed_w: the reversed latent, we want to detect the watermark in it
        :return: the probability of the watermark existing in the latent
        """
        posteriors = self.recover_posteriors(reversed_w)
        generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, noise_rate, test_bits, g, max_bp_iter, t = self.decoding_key
        fpr = self.fpr

        posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)

        r = parity_check_matrix.shape[0]
        Pi = np.prod(posteriors[parity_check_matrix.indices.reshape(r, t)], axis=1)
        log_plus = np.log((1 + Pi) / 2)
        log_minus = np.log((1 - Pi) / 2)
        log_prod = log_plus + log_minus

        const = 0.5 * np.sum(np.power(log_plus, 2) + np.power(log_minus, 2) - 0.5 * np.power(log_prod, 2))
        threshold = np.sqrt(2 * const * np.log(1 / fpr)) + 0.5 * log_prod.sum()

        self.num += 1
        if log_plus.sum() >= threshold:
            self.tp += 1

        diff = log_plus.sum() - threshold
        return 1 / (1 + np.exp(-diff))



    def get_tpr(self):
        """
        get the true positive rate.
        the state should be maintained via detect function
        :return:  the true positive rate
        """
        return self.tp / self.num if self.num != 0 else 0

    def KeyGen(self, g=None, r=None, noise_rate=None):
        # Set basic scheme parameters
        num_test_bits = int(np.ceil(np.log2(1 / self.fpr)))
        secpar = int(np.log2(binom(self.latent_length, self.prc_t)))
        if g is None: g = secpar
        if noise_rate is None: noise_rate = 1 - 2 ** (-secpar / g ** 2)
        k = self.message_length + g + num_test_bits
        if r is None: r = self.latent_length - k - secpar

        # Sample n by k generator matrix (all but the first n-r of these will be over-written)
        generator_matrix = self.GF.Random((self.latent_length, k))

        # Sample scipy.sparse parity-check matrix together with the last n-r rows of the generator matrix
        row_indices = []
        col_indices = []
        data = []
        for row in range(r):
            chosen_indices = np.random.choice(self.latent_length - r + row, self.prc_t - 1, replace=False)
            chosen_indices = np.append(chosen_indices, self.latent_length - r + row)
            row_indices.extend([row] * self.prc_t)
            col_indices.extend(chosen_indices)
            data.extend([1] * self.prc_t)
            generator_matrix[self.latent_length - r + row] = generator_matrix[chosen_indices[:-1]].sum(axis=0)
        parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))

        # Compute scheme parameters
        max_bp_iter = int(np.log(self.latent_length) / np.log(self.prc_t))

        # Sample one-time pad and test bits
        one_time_pad = self.GF.Random(self.latent_length)
        test_bits = self.GF.Random(num_test_bits)

        # Permute bits
        permutation = np.random.permutation(self.latent_length)
        generator_matrix = generator_matrix[permutation]
        one_time_pad = one_time_pad[permutation]
        parity_check_matrix = parity_check_matrix[:, permutation]

        encoding_key = (generator_matrix, one_time_pad, test_bits, g, noise_rate)
        decoding_key = (
        generator_matrix, parity_check_matrix, one_time_pad, self.fpr, noise_rate, test_bits, g, max_bp_iter,
        self.prc_t)

        return encoding_key, decoding_key

    def encode(self, message=None):
        """
        :param message: 01 bits array, len(message) <= self.message_length
        :return: the codeword
        """
        generator_matrix, one_time_pad, test_bits, g, noise_rate = self.get_key()['encoding_key']
        n, k = generator_matrix.shape

        if message is None:
            random_g = self.GF.Random(g)
            random_msg = self.GF.Random(self.message_length)
            padding = self.GF.Zeros(k - len(test_bits) - g - len(random_msg))
            payload = np.concatenate((test_bits, random_g, random_msg, padding))
            self.message = random_msg
        else:
            assert len(message) <= k - len(test_bits) - g, "Message is too long"
            self.message = message
            payload = np.concatenate(
                (test_bits, self.GF.Random(g), self.GF(message), self.GF.Zeros(k - len(test_bits) - g - len(message))))

        error = self.GF(np.random.binomial(1, noise_rate, n))

        return 1 - 2 * torch.tensor(payload @ generator_matrix.T + one_time_pad + error, dtype=float)


    def sample(self, codeword):
        codeword_np = codeword.numpy()
        pseudogaussian_np = codeword_np * np.abs(np.random.randn(*codeword_np.shape))
        pseudogaussian = torch.from_numpy(pseudogaussian_np).to(dtype=torch.float64)
        if self.basis is None:
            return pseudogaussian
        return pseudogaussian @ self.basis.T

    def recover_posteriors(self, z):
        denominators = np.sqrt(2 * self.var * (1 + self.var))

        if self.basis is None:
            return erf(z / denominators)
        else:
            return erf((z @ self.basis) / denominators)




