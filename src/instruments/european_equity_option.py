import numpy as np
from scipy.stats import norm
from src.call_or_put import CallOrPut
from src.long_or_short import LongOrShort
from src.monte_carlo_results import MonteCarloResults
from src.gbm.time_dependent_gbm import TimeDependentGBM
from src.gbm.time_independent_gbm import TimeIndependentGBM


class EuropeanEquityOption:
    """
    A class representing a European Equity Option.
    """

    def __init__(
            self,
            notional: float,
            initial_spot: float,
            strike: float,
            interest_rate: float,
            volatility: float,
            time_to_maturity: float,
            call_or_put: CallOrPut,
            long_or_short: LongOrShort):
        """
        Class constructor.

        :param notional: Notional.
        :param initial_spot: Initial equity spot.
        :param strike: Equity strike.
        :param interest_rate: Interest rate.
        :param volatility: Volatility.
        :param time_to_maturity: Time to maturity.
        :param call_or_put: Indicates if the option is a 'CALL' or a 'PUT'.
        :param long_or_short: Indicates if the option is 'LONG' or 'SHORT'.
        """
        self.notional: float = notional
        self.initial_spot: float = initial_spot
        self.strike: float = strike
        self.interest_rate: float = interest_rate
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

    def get_black_scholes_price(self) -> float:
        """
        Gets the analytical Black-Scholes price for the European equity option.

        :return: The analytical Black-Scholes price.
        """
        initial_spot: float = self.notional * self.initial_spot
        strike: float = self.notional * self.strike

        d_1: float = \
            (np.log(initial_spot / strike) +
             ((self.interest_rate + 0.5 * self.volatility**2) * self.time_to_maturity)) / \
            (self.volatility * np.sqrt(self.time_to_maturity))

        d_2: float = d_1 - self.volatility * np.sqrt(self.time_to_maturity)
        direction: float = 1 if self.long_or_short == LongOrShort.LONG else -1

        if self.call_or_put == CallOrPut.CALL:
            return direction * \
                   (initial_spot * norm.cdf(d_1) -
                    strike * np.exp(-1 * self.interest_rate * self.time_to_maturity) * norm.cdf(d_2))

        elif self.call_or_put == CallOrPut.PUT:
            return direction * \
                   (-1 * initial_spot * norm.cdf(-1 * d_1) +
                    strike * np.exp(-1 * self.interest_rate * self.time_to_maturity) * norm.cdf(-1 * d_2))

    def get_time_independent_monte_carlo_price(
            self,
            number_of_paths: int,
            number_of_time_steps: int,
            plot_paths: bool = False,
            show_stats: bool = False) -> MonteCarloResults:
        """
        Returns the price for a 'CALL' or 'PUT' equity european option using monte carlo simulations where the
        volatility is time-independent.

        :param number_of_paths: Number of paths in the Monte Carlo simulation.
        :param number_of_time_steps: Number of time steps in the Monte Carlo simulation.
        :param plot_paths: Plot the simulated paths from the Monte Carlo simulation. Default = False.
        :param show_stats: Show statistics for the simulated paths from the Monte Carlo simulation. Default = False.
        :return: The price for a 'CALL' or 'PUT' equity european option using monte carlo simulations where the
        volatility is time-independent.
        """

        gbm: TimeIndependentGBM = \
            TimeIndependentGBM(
                drift=self.interest_rate,
                volatility=self.volatility,
                initial_spot=self.initial_spot,
                time_to_maturity=self.time_to_maturity)

        paths: np.ndarray = \
            self.notional * gbm.get_paths(
                number_of_paths=number_of_paths,
                number_of_time_steps=number_of_time_steps)

        if plot_paths:
            gbm.create_plots(paths)

        if show_stats:
            gbm.get_path_statistics(paths)

        if self.call_or_put == CallOrPut.CALL:
            payoffs = \
                np.maximum(paths[:, -1] - self.notional * self.strike, 0) * \
                np.exp(-1 * self.interest_rate * self.time_to_maturity)

            price: float = np.average(payoffs)
            error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
            return MonteCarloResults(price, error)

        elif self.call_or_put == CallOrPut.PUT:
            payoffs = \
                np.maximum(self.notional * self.strike - paths[:, -1], 0) * \
                np.exp(-1 * self.interest_rate * self.time_to_maturity)

            price: float = np.average(payoffs)
            error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
            return MonteCarloResults(price, error)

    def get_time_dependent_monte_carlo_price(
            self,
            number_of_paths: int,
            number_of_time_steps: int,
            volatility_excel_path: str,
            volatility_excel_sheet_name: str,
            plot_paths: bool = False,
            show_stats: bool = False):
        """
        Returns the price for a 'CALL' or 'PUT' equity european option using monte carlo simulations where the
        volatility is time-dependent.

        :param number_of_paths: Number of paths in the Monte Carlo simulation.
        :param number_of_time_steps: Number of time steps in the Monte Carlo simulation
        :param volatility_excel_path: Specifies the path where the Excel file of volatilities is stored.
        :param volatility_excel_sheet_name: Specifies the name of the Excel sheet where the volatilities are stored.
        :param plot_paths: Plot the simulated paths from the Monte Carlo simulation. Default = False.
        :param show_stats: Show statistics for the simulated paths from the Monte Carlo simulation. Default = False
        :return: The price for a 'CALL' or 'PUT' equity european option using monte carlo simulations where the
        volatility is time-dependent.
        """

        gbm: TimeDependentGBM = \
            TimeDependentGBM(
                drift=self.interest_rate,
                excel_file_path=volatility_excel_path,
                sheet_name=volatility_excel_sheet_name,
                initial_spot=self.initial_spot,
                time_to_maturity=self.time_to_maturity)

        paths: np.ndarray = \
            self.notional * \
            gbm.get_gbm_paths(number_of_paths, number_of_time_steps, self.initial_spot, self.time_to_maturity)

        if plot_paths:
            gbm.create_plots(paths)

        if show_stats:
            gbm.get_path_statistics(paths)

        if self.call_or_put == CallOrPut.CALL:
            payoffs = \
                np.maximum(paths[:, -1] - self.notional * self.strike, 0) * \
                np.exp(-1 * self.interest_rate * self.time_to_maturity)

            price: float = np.average(payoffs)
            error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
            return MonteCarloResults(price, error)

        elif self.call_or_put == CallOrPut.PUT:
            payoffs = \
                np.maximum(self.notional * self.strike - paths[:, -1], 0) * \
                np.exp(-1 * self.interest_rate * self.time_to_maturity)

            price: float = np.average(payoffs)
            error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
            return MonteCarloResults(price, error)
