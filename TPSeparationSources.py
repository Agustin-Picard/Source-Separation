import numpy as np
import scipy.io.wavfile
import scipy as scp
import matplotlib.pyplot as plt
from functools import reduce
from math import tanh, cosh, sqrt
from itertools import product

"""Developed by Agustin M. Picard and Tomás Volker as a project for 
a Signal Processing course at IMT Atlantique"""

class SignalEnsemble(object):
    def __init__(self, values):
        self.channel = np.array(values)
        self.channel_count = values.shape[0]
        self.duration = values.shape[1]
        self.time_steps = np.arange(0, self.duration)

    def mix(self, coefficients):
        """
        Does the mixing of the channels according to the coefficients matrix/array
        :param coefficients: mixing coefficients in an array/matrix
        :return: SignalEnsemble with the resulting channels as inputs
        """
        return SignalEnsemble(np.dot(coefficients, self.channel))

    @staticmethod
    def colinearity_factor(v: np.ndarray, u: np.ndarray):
        """
        Calculates the linear algebra colinearity factor for two 1D arrays
        :param v: input array
        :param u: input array
        :return: scalar in the range [-1, 1] that indicates the angle between the vectors
        """
        return np.inner(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    def decorrelate(self):
        """
        Whitens the channels thanks to the eigenvalue/vector representation
        :return: SignalEnsemble with the decorrelated/whitened signals
        """
        return self.mix(self.calculate_whitening())

    def calculate_whitening(self):
        """
        Performs a whitening through the eigenvalue/vector representation
        :return: matrix with the mixing coefficients for the whitening
        """
        (D, E) = np.linalg.eig(self.correlation_matrix())
        D = np.diag(D ** (-1 / 2))
        sqrtCinv = np.dot(E, np.dot(D, E.T))

        return sqrtCinv

    def correlation_matrix(self, delay=0) -> np.ndarray:
        """
        Calculates the correlation matrix for the n channels with itself or
        delayed self
        :param delay: amount of delay
        :return: n x n matrix with the correlation coefficients
        """
        cx = np.zeros((self.channel_count, self.channel_count))

        if delay is not 0:
            delayed_signal = np.roll(self.channel, delay, axis=1)
        else:
            delayed_signal = self.channel

        for i, j in product(range(self.channel_count), range(self.channel_count)):
            cx[i, j] = reduce(lambda x, y: x + y,
                              map(lambda x, y: x * y, self.channel[i], delayed_signal[j])) / self.duration

        # Equivalent to this matrix product but it threw memory error...
        # cx = np.dot(self.channel, delayed_signal.T) / self.duration

        return cx

    def plot_results(self):
        """
        Plots the 2 channels (the example has 2 channels)
        """
        plt.figure('Signals')
        plt.subplot(2, 1, 1)
        plt.plot(self.channel[0])
        plt.subplot(2, 1, 2)
        plt.plot(self.channel[1])
        plt.show()

    def normalize(self):
        """
        Normalizes the channels' amplitude
        """
        for i in range(self.channel_count):
            channel_max = np.max(self.channel[i])
            self.channel[i] = self.channel[i] / channel_max

    @staticmethod
    def tanh(x) -> np.ndarray:
        return np.array(list(map(tanh, x)))

    @staticmethod
    def sech2(x) -> np.ndarray:
        return np.array(list(map(lambda z: 1.0 / (cosh(z) ** 2), x)))


class SignalEnsembleICA(SignalEnsemble):
    """Signal Ensemble implementing the ICA channel separator"""
    def __init__(self, values):
        SignalEnsemble.__init__(self, values)

    def ICA(self, whitened=True):
        """
        Performs the whole ICA algorithm for channel separatiom
        :param whitened: boolean indicating whether the channels have been whitened
        :return: SignalEnsembleICA with the separated channels
        """
        if whitened is not True:
            c = self.calculate_whitening()
            self.channel = np.dot(c, self.channel)
        w = self.independent_component_analysis()
        w = self.orthogonalize(w)

        return SignalEnsemble.mix(self, w)

    def independent_component_analysis(self) -> np.ndarray:
        """
        Estimates the channel separation matrix
        :return: non orthogonalized channel separation matrix
        """
        comps = []

        while len(comps) < self.channel_count:
            c = self.retrieve_independent_component()
            base = list(filter(lambda x: np.abs(self.colinearity_factor(c, x)) < 0.25, comps))
            if len(comps) == len(base) or len(comps) == 0:
                comps.append(c)

        return np.array(comps)

    def retrieve_independent_component(self, it=10) -> np.ndarray:
        """
        Estimates a channel separation vector
        :param it: amount of iterations to run the Newton-Raphson algorithm
        :return: array with the new optimal vector
        """
        w = np.random.randn(self.channel_count)

        for i in range(it):
            w_x = np.dot(w.T, self.channel)
            w_next = np.dot(self.channel, self.tanh(w_x)) / self.duration - np.mean(self.sech2(w_x)) * w
            w_next /= np.linalg.norm(w_next)
            w = w_next

        return w

    def orthogonalize(self, w: np.ndarray) -> np.ndarray:
        """
        Orthogonalizes the channel separation matrix using a Gram-Schmidt scheme
        :param w: target matrix
        :return: orthogonal matrix
        """
        r = [w[0]]

        for i in range(1, w.shape[0]):
            r.append(w[i] - sum(r, lambda x: self.colinearity_factor(w[i], x) * x))

        return np.array(r)


class SignalEnsembleSimpleSOBI(SignalEnsemble):
    """SignalEnsemble class implementing the simple SOBI algorithm"""
    def __init__(self, values):
        SignalEnsemble.__init__(self, values)

    def simple_SOBI(self):
        """
        Simple SOBI implementation taking advantage of the eigenvalue
        decomposition with unitary delay
        :return: SignalEnsembleSOBI object with separated channels
        """
        (D, E) = np.linalg.eig(self.correlation_matrix(delay=1))

        return SignalEnsemble.mix(self, E.T)


class SignalEnsembleSOBI(SignalEnsemble):
    """Signal Ensemble class that implements the SOBI algorithm"""
    def __init__(self, values):
        SignalEnsemble.__init__(self, values)

    def SOBI(self, win=6):
        """
        Implementation of the SOBI algorithm
        :param win: window over which the Rs are calculated
        :return: SignalEnsemble with the separated sources
        """
        assert self.channel_count == 2
        T1 = self.calculate_tr((0, 0))
        T2 = self.calculate_tr((1, 1))
        T12 = self.calculate_tr((0, 1))

        F1 = self.calculate_off((0, 0), win)
        F2 = self.calculate_off((1, 1), win)
        F12 = self.calculate_off((0, 1), win)

        A = self.calculate_A((T1, T2, T12), (F1, F2, F12))

        return SignalEnsemble.mix(self, np.linalg.pinv(A))

    def calculate_correlation(self, elems: tuple, delta: int):
        """
        Calculates the correlation in time between the selected channels with a specific
        delay
        :param elems: selected channels
        :param delta: delay over which the correlation is going to be calculated
        :return: total correlation
        """
        available_values = self.duration - delta
        return sum(
            range(available_values),
            lambda t: self.channel[elems[0]][t] * self.channel[elems[1]][t + delta]
        ) / available_values

    def calculate_tr(self, elems: tuple) -> float:
        """
        Calculates the trace of the matrix of the correlation matrix for the
        selected channels
        :param elems: selected channels
        :return: corresponding trace
        """
        return self.calculate_correlation(elems, delta=0)

    def calculate_off(self, elems: tuple, win: int):
        """
        Calculates the sum over the non-diagonal elements of the
        correlation matrix for the selected channels over with chosen
        window
        :param elems: selected channels
        :param win: selected window
        :return: corresponding off of the matrix
        """
        upper_average = sum(
            range(1, win),
            lambda t: (win - t) * self.calculate_correlation(elems, t)
        ) / (0.5 * win * (win - 1))

        if elems[0] == elems[1]:
            return upper_average

        lower_average = sum(
            range(1, win),
            lambda t: (win - t) * self.calculate_correlation((elems[1], elems[0]), t)
        ) / (0.5 * win * (win - 1))

        return (upper_average + lower_average) / 2

    @staticmethod
    def calculate_A(T: tuple, F: tuple) -> np.ndarray:
        """
        Calculates the A matrix for the tuple of traces and
        off matrix values
        :param T: traces for both channels
        :param F: off values for both channels
        :return: A matrix for obtaining the separating mix
        """
        T1, T2, T12 = T
        F1, F2, F12 = F

        alpha = 2 * F12 * T12 - F1 * T2 - F2 * T1
        beta = 2 * (T12**2 - T1 * T2)
        gamma2 = (F1 * T2 - F2 * T1) ** 2 + 4 * (F12 * T2 - T12 * F2) * (F12 * T1 - T12 * F1)
        d1, d2 = alpha - sqrt(gamma2), alpha + sqrt(gamma2)
        return np.array([[beta * F1 - T1 * d1, beta * F12 - T12 * d2],
                         [beta * F12 - T12 * d1, beta * F2 - T2 * d2]])


class StochasticSignalEnsemble(object):
    """Signal Ensemble for the first part of the project"""
    def __init__(self, n=2, k=1000):
        self.channel = np.zeros((n, k))
        self.channel_count = n
        self.duration = k

    def generate_uniform(self, win=sqrt(3)):
        """
        Generates the stochastic uniform signals for the specified duration and
        window
        :param win: window for the uniform distribution (assuming symmetric)
        """
        self.channel = np.random.uniform(-win, win, self.channel.shape)

    def generate_gaussian(self, std=1):
        """
        Generates the stochastic gaussian signals for the specified duration
        and standard deviation
        :param std: specified standard deviation
        """
        for i in range(self.channel_count):
            self.channel[i] = std * np.random.randn(self.duration)

    def correlate_channels(self, c: np.ndarray):
        """
        Correlated the channels with the specified c matrix
        as a mix
        :param c: matrix for mixing the channels
        """
        self.channel = np.dot(self.channel.T, c).T

    def scatter_plot(self):
        """
        Does the scatter plot of the two channels
        """
        assert self.channel_count == 2
        plt.figure('Stochastic Signals Scatter Plot')
        plt.scatter(self.channel[0], self.channel[1])
        plt.show()


def sum(range, selector):
    result = 0

    for i in range:
        result += selector(i)

    return result


def read_input(filename: str) -> np.ndarray:
    """
    Reads the x64 binary file
    :param filename: name of the file
    :return: n-dimensional array with the values from
    the file
    """
    return np.fromfile(filename, dtype=np.float64)


def normalized_quadratic_error(ref: np.ndarray, signal: np.ndarray):
    """
    Calculates the EQMN for a reference and a filtered signals
    :param ref: reference signal
    :param signal: filtered signal
    :return: EQMN value
    """
    err = np.inner(ref, signal) / (np.linalg.norm(ref) * np.linalg.norm(signal))
    return err


def to_dB(x):
    """
    Converts a float value to dB scale
    :param x: float value to be converted
    :return: dB value of the corresponding float
    """
    return 10 * np.log10(1 - x**2)


def convert_to_wav(data: np.ndarray, filename: str):
    """
    Converts the data array to a wav file
    :param data: data array to be converted
    :param filename: name of the wav file
    """
    scp.io.wavfile.write(filename, rate=44100, data=data)


def plot_signal_comparison(references: tuple, results: np.ndarray):
    """
    Plots the 2 reference sources and their filtered signal counterparts
    :param references:
    :param results:
    """
    plt.figure('Signal Comparison')
    plt.subplot(4, 1, 1)
    plt.title('Reference 1')
    plt.plot(references[0])
    plt.grid()
    plt.subplot(4, 1, 2)
    plt.title('Reference 2')
    plt.plot(references[1])
    plt.grid()
    plt.subplot(4, 1, 3)
    plt.title('Filtered Signal 1')
    plt.plot(results[0])
    plt.grid()
    plt.subplot(4, 1, 4)
    plt.title('Filtered Signal 2')
    plt.plot(results[1])
    plt.grid()
    plt.show()


def main():
    # Start with the stochastic signals
    stochastic_signals = StochasticSignalEnsemble()

    # Generate the uniforms
    stochastic_signals.generate_uniform()
    stochastic_signals.scatter_plot()

    # Correlate them
    c = np.array([[1, 1], [-1, 2]])
    stochastic_signals.correlate_channels(c)
    stochastic_signals.scatter_plot()

    # Now with the gaussians
    stochastic_signals.generate_gaussian()
    stochastic_signals.scatter_plot()

    # Correlate them
    stochastic_signals.correlate_channels(c)
    stochastic_signals.scatter_plot()

    # Read the input files
    in_1 = read_input('In_1.txt')
    in_2 = read_input('In_2.txt')

    # Read the references' files
    ref_1 = read_input('Ref_1.txt')
    ref_2 = read_input('Ref_2.txt')

    # Define the ensemble and mix the signals
    measurements = SignalEnsemble(np.vstack((in_1, in_2)))

    # Perform the ICA algorithm
    decorrelated = measurements.decorrelate()
    output_ICA = SignalEnsembleICA(decorrelated.channel)
    output_ICA = output_ICA.ICA()
    output_ICA.normalize()

    # Now, the simple SOBI algorithm for a unitary delay unit
    output_simple_SOBI = SignalEnsembleSimpleSOBI(decorrelated.channel)
    output_simple_SOBI = output_simple_SOBI.simple_SOBI()
    output_simple_SOBI.normalize()

    # Now perform the actual SOBI algorithm
    output_SOBI = SignalEnsembleSOBI(measurements.channel)
    output_SOBI = output_SOBI.SOBI()
    output_SOBI.normalize()

    # Plot the outputs and references for comparison
    plot_signal_comparison((ref_1, ref_2), output_ICA.channel)
    plot_signal_comparison((ref_1, ref_2), output_simple_SOBI.channel)
    plot_signal_comparison((ref_1, ref_2), output_SOBI.channel)

    # Convert the arrays to .wav files to hear the results
    convert_to_wav(ref_1, "Ref_1.wav")
    convert_to_wav(ref_2, "Ref_2.wav")
    convert_to_wav(in_1, "In_1.wav")
    convert_to_wav(in_2, "In_2.wav")
    convert_to_wav(output_ICA.channel[0], "ICA_1.wav")
    convert_to_wav(output_ICA.channel[1], "ICA_2.wav")
    convert_to_wav(output_SOBI.channel[0], "SOBI_1.wav")
    convert_to_wav(output_SOBI.channel[1], "SOBI_2.wav")

    # Print the normalized quadratic error to measure the quality of each of the algorithms
    print('ICA 1: {0} | ICA 2: {1} | SOBI 1: {2} | SOBI 2: {3}'.format(
        to_dB(normalized_quadratic_error(ref_1, output_ICA.channel[0])),
        to_dB(normalized_quadratic_error(ref_2, output_ICA.channel[1])),
        to_dB(normalized_quadratic_error(ref_1, output_SOBI.channel[0])),
        to_dB(normalized_quadratic_error(ref_2, output_SOBI.channel[1]))))


if __name__ == '__main__':
    main()
