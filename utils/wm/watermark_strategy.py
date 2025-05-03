import typing
from abc import ABC, abstractmethod
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.special import binom
import galois
from scipy.special import erf
from ldpc import bp_decoder
import math
from utils.wm.wm_provider import WmProvider



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





### Given a GF(2) matrix, do row elimination and return the first k rows of A that form an invertible matrix
def boolean_row_reduce(A, print_progress=False):
    n, k = A.shape
    A_rr = A.copy()
    perm = np.arange(n)
    for j in range(k):
        idxs = j + np.nonzero(A_rr[j:, j])[0]
        if idxs.size == 0:
            print("The given matrix is not invertible")
            return None
        A_rr[[j, idxs[0]]] = A_rr[[idxs[0], j]]  # For matrices you have to swap them this way
        (perm[j], perm[idxs[0]]) = (perm[idxs[0]], perm[j])  # Weirdly, this is MUCH faster if you swap this way instead of using perm[[i,j]]=perm[[j,i]]
        A_rr[idxs[1:]] += A_rr[j]
    if print_progress: print()
    return perm[:k]





class PRCWatermark(WatermarkStrategy, WmProvider):
    def __init__(self,
                 fpr=0.00001,
                 prc_t=3,
                 letent_length=16384,
                 message_length=512,
                 var=1.5,
                 basis=None,
                 **kwargs
                 ):
        WmProvider.__init__(self, **kwargs)
        self.fpr = fpr
        self.prc_t = prc_t
        self.latent_length = letent_length  # 4 * 64 * 64
        self.encoding_key, self.decoding_key = None, None
        self.GF = galois.GF(2)
        self.message_length = message_length
        self.var = var # for decoding
        self.basis = basis # for decoding / sampling
        self.tp = 0
        self.num = 0

        ### will refresh for each call
        self.message = None
        self.messages = None

        assert math.prod(
            self.latent_shape) == self.latent_length, f"latent_shape {self.latent_shape} is not consistent with latent_length {self.latent_length}"





    """
    WMProvider interface
    """

    def get_wm_type(self) -> str:
        return "PRC"

    def wiggle_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Resample latents

        @param latents: latent tensor with batch dim
        @return: torch.Tensor with batch dim
        """

        # reverse sampling back to barcode pixels in [0, 2**self.l - 1]
        latents = latents.detach().cpu().numpy()
        latents = norm.cdf(latents) * 2 ** self.l
        latents = latents.astype(np.int32)
        # fix bug where we sometimes get 2**l
        latents[latents == 2 ** self.l] = 2 ** self.l - 1
        # latents is now integers in [0, 2**self.l - 1]

        # forward sampling with randomnes
        # y we already have
        y = latents
        # u we draw
        u = np.random.uniform(low=0, high=1, size=y.shape).astype(np.float32)
        # sampling a gaussian
        new_latent = norm.ppf((u + y) / 2 ** self.l)

        return torch.tensor(new_latent, dtype=self.dtype, device=self.device)

    def get_wm_latents(self, **kwargs):
        """
        Get Watermarked latents and barcodes

        @return: dict
        """
        latents_torch = []
        message_bits_str_list = []
        messages = []
        for _ in range(0, self.batch_size):
            latent_torch = self.get_init_latent(dim=self.latent_shape).to(self.device)
            # remember the message bits as string
            message_bits_str_list.append(''.join(str(int(bit)) for bit in self.message))
            messages.append(self.message)

            latents_torch.append(latent_torch.squeeze(0))

        # finalize
        latents_torch = torch.stack(latents_torch, dim=0)

        results_dict = {"zT_torch": latents_torch.float(),
                        "message_bits_str_list": message_bits_str_list
                        }
        self.messages = messages
        return results_dict

    def get_accuracies(self, latents: typing.Union[torch.Tensor, np.array]) -> typing.Dict[str, any]:
        """
        Get bit accuracy between original and extracted messages

        @param latents: latent either tensor with batch dim or numpy with batch dim
        @return: dict
        """

        # iterate and calulate bit accuracies
        bit_accuracies = []
        recovered_message_bits_str_list = []

        for i in range(0, self.batch_size):
            #print(f"bit accuracy: {bit_accuracy}")
            #print(f"message: {self.message}")
            #print(f"latent: {latents[i]}")

            posteriors = self.recover_posteriors(latents[i].flatten().cpu())
            msg_numpy_array = self.decode(posteriors)

            if msg_numpy_array is not None:
                bit_acc = (msg_numpy_array[:len(self.messages[i])] == self.messages[i]).sum() / len(self.messages[i])
            else:
                bit_acc = 0

            bit_accuracies.append(bit_acc)
            if msg_numpy_array is None:
                recovered_message_bits_str_list.append(None)
            else:
                recovered_message_bits_str_list.append(''.join(str(int(bit)) for bit in msg_numpy_array))

        return {
            "accuracies": bit_accuracies,
            "bit_accuracies": bit_accuracies,
            "message_bits_str_list": recovered_message_bits_str_list
        }



    """
    Strategy interface
    """

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


    def get_init_latent(self, dim = (1, 4, 64, 64), message = None):
        """
        return the initial latent vector, the shape is consistent by the actual impl(e.g. [1, 4, 64, 64])
        :return: some tensor on cpu
        """
        prc_codeword = self.encode(message)
        return self.sample(prc_codeword).reshape(*dim)


    def detect(self, reversed_w):
        """
        detect if the watermark exists in the latent
        :param reversed_w: the reversed latent, we want to detect the watermark in it
        :return: the probability of the watermark existing in the latent
        """
        posteriors = self.recover_posteriors(reversed_w.flatten())

        recovered_message = self.decode(posteriors)

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

        # bit accuracy
        if recovered_message is not None and self.message is not None:
            bit_acc = (recovered_message[:len(self.message)] == self.message).sum() / len(self.message)
        else:
            bit_acc = 0

        return bit_acc



    def get_tpr(self):
        """
        get the true positive rate.
        the state should be maintained via detect function
        :return:  the true positive rate
        """
        return self.tp / self.num if self.num != 0 else 0



    """
    PRC Watermark
    Borrowed from https://github.com/XuandongZhao/PRC-Watermark
    """

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
        :param message: numpy 01 bits array, len(message) <= self.message_length
        :return: the codeword
        """
        generator_matrix, one_time_pad, test_bits, g, noise_rate = self.get_key()['encoding_key']
        n, k = generator_matrix.shape

        if message is None:
            random_g = self.GF.Random(g)
            random_msg = self.GF.Random(self.message_length)
            padding = self.GF.Zeros(k - len(test_bits) - g - len(random_msg))
            payload = np.concatenate((test_bits, random_g, random_msg, padding))
            self.message = np.asarray(random_msg, dtype=int)
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

    ### Decoder
    ## Inputs:
    # decoding_key - Decoding key output by KeyGen.
    # posteriors - The posterior expectations of sign(z) as a torch.tensor.
    ## Returns:
    # recovered_message - The recovered message. If the test bits are incorrect, outputs None.
    def decode(self, posteriors):
        generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, noise_rate, test_bits, g, max_bp_iter_key, t = self.decoding_key

        max_bp_iter = max_bp_iter_key

        posteriors2 = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)

        channel_probs = (1 - np.abs(posteriors2)) / 2
        x_recovered = (1 - np.sign(posteriors2)) // 2


        bpd = bp_decoder(parity_check_matrix, channel_probs=channel_probs, max_iter=max_bp_iter,
                         bp_method="product_sum")
        x_decoded = bpd.decode(x_recovered)

        # Compute a confidence score.
        bpd_probs = 1 / (1 + np.exp(bpd.log_prob_ratios))
        confidences = 2 * np.abs(0.5 - bpd_probs)

        # Order codeword bits by confidence.
        confidence_order = np.argsort(-confidences)
        ordered_generator_matrix = generator_matrix[confidence_order]
        ordered_x_decoded = x_decoded[confidence_order]

        # Find the first (according to the confidence order) linearly independent set of rows of the generator matrix.
        top_invertible_rows = boolean_row_reduce(ordered_generator_matrix)
        if top_invertible_rows is None:
            return None

        # Solve the system.
        recovered_string = np.linalg.solve(ordered_generator_matrix[top_invertible_rows],
                                           self.GF(ordered_x_decoded[top_invertible_rows]))

        if not (recovered_string[:len(test_bits)] == test_bits).all():
            return None
        return np.array(recovered_string[len(test_bits) + g:])




