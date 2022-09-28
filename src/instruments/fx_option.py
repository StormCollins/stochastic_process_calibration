import numpy as np
from scipy.stats import norm
from src.call_or_put import CallOrPut
from src.long_or_short import LongOrShort
from src.monte_carlo_results import MonteCarloResults
from src.gbm.time_independent_gbm import TimeIndependentGBM


class FxOption:
    """
    A class representing an FX (Foreign Exchange) option.
    """

    def __init__(
            self,
            notional: float,
            initial_spot: float,
            strike: float,
            domestic_interest_rate: float,
            foreign_interest_rate: float,
            volatility: float,
            time_to_maturity: float,
            call_or_put: CallOrPut,
            long_or_short: LongOrShort):
        self.notional: float = notional
        self.initial_spot: float = initial_spot
        self.strike: float = strike
        self.domestic_interest_rate: float = domestic_interest_rate
        self.foreign_interest_rate: float = foreign_interest_rate
        self.volatility: float = volatility
        self.time_to_maturity: float = time_to_maturity

        if call_or_put not in [CallOrPut.CALL, CallOrPut.PUT]:
            raise ValueError(f'Unknown option type: {call_or_put}')
        else:
            self.call_or_put: CallOrPut = call_or_put

        if long_or_short not in [LongOrShort.LONG, LongOrShort.SHORT]:
            raise ValueError(f'Unknown direction: {long_or_short}')
        else:
            self.long_or_short: LongOrShort = long_or_short

    def get_garman_kohlhagen_price(self) -> float:
        """
        The Garman-Kohlahgen model is an analytical model for valuing European options on foreign exchange.
        It's a modification to the Black-Scholes model such that the model can deal with two interest rates, the
        domestic interest rate and the foreign interest rate.

        :return: Garman_kohlhagen price for an FX option.
        """
        initial_spot: float = self.notional * self.initial_spot
        strike: float = self.notional * self.strike
        drift: float = self.domestic_interest_rate - self.foreign_interest_rate

        d_1: float = \
            (np.log(initial_spot / strike) + ((drift + 0.5 * self.volatility ** 2) * self.time_to_maturity)) / \
            (self.volatility * np.sqrt(self.time_to_maturity))

        d_2: float = d_1 - self.volatility * np.sqrt(self.time_to_maturity)
        direction: float = 1 if self.long_or_short == LongOrShort.LONG else -1

        if self.call_or_put == CallOrPut.CALL:
            return direction * \
                   (initial_spot * np.exp(-1 * self.foreign_interest_rate * self.time_to_maturity) * norm.cdf(d_1) -
                    self.strike * np.exp(-1 * self.domestic_interest_rate * self.time_to_maturity) * norm.cdf(d_2))

        elif self.call_or_put == CallOrPut.PUT:
            return direction * \
                   (-1 * initial_spot * np.exp(-1 * self.foreign_interest_rate * self.time_to_maturity) *
                    norm.cdf(-1 * d_1) +
                    self.strike * np.exp(-1 * self.domestic_interest_rate * self.time_to_maturity) * norm.cdf(-1 * d_2))

    def get_time_independent_monte_carlo_price(
            self,
            number_of_paths: int,
            number_of_time_steps: int,
            plot_paths: bool = True,
            show_stats: bool = False) -> MonteCarloResults:
        """
        Returns the price for a 'CALL' or 'PUT' FX option using monte carlo simulations.

        :param show_stats: If set to TruDisplays the mean, standard deviation, 95% PFE and normality test.
        :param number_of_paths: Number of current_value for the FX option.
        :param number_of_time_steps: Number of time steps for the FX option.
        :param plot_paths: If set to True plots the current_value.
        :return: Monte Carlo price for an FX Option.
        """
        drift: float = self.domestic_interest_rate - self.foreign_interest_rate

        gbm: TimeIndependentGBM = \
            TimeIndependentGBM(drift, self.volatility, self.notional, self.initial_spot, self.time_to_maturity)

        paths: np.ndarray = gbm.get_paths(number_of_paths, number_of_time_steps)

        if plot_paths:
            gbm.create_plots(paths)

        if show_stats:
            gbm.get_path_statistics(paths)

        if self.call_or_put == CallOrPut.CALL:
            payoffs = \
                np.maximum(paths[:, -1] - self.notional * self.strike, 0) * \
                np.exp(-1 * self.domestic_interest_rate * self.time_to_maturity)

            price: float = np.average(payoffs)
            error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
            return MonteCarloResults(price, error)

        elif self.call_or_put == CallOrPut.PUT:
            payoffs = \
                np.maximum(self.notional * self.strike - paths[:, -1], 0) * \
                np.exp(-1 * self.domestic_interest_rate * self.time_to_maturity)

            price: float = np.average(payoffs)
            error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
            return MonteCarloResults(price, error)
