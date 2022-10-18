"""
Contains a class for representing an FX (Foreign Exchange) forward.
"""
import numpy as np
from scipy.stats import norm
from src.enums_and_named_tuples.monte_carlo_pricing_results import MonteCarloPricingResults
from src.enums_and_named_tuples.long_or_short import LongOrShort
from src.gbm.time_dependent_gbm import TimeDependentGBM
from src.gbm.time_independent_gbm import TimeIndependentGBM


class FxForward:
    """
    A class representing an FX (Foreign Exchange) forward.
    We follow the convention of quoting domestic (DOM) currency per foreign (FOR) currency i.e., FORDOM
    e.g., USDZAR = 17, means ZAR is the domestic currency, USD is the foreign currency, and it is 17 domestic (ZAR) to
    1 foreign (USD).
    """

    def __init__(
            self,
            notional: float,
            initial_spot: float,
            strike: float,
            domestic_interest_rate: float,
            foreign_interest_rate: float,
            time_to_maturity: float,
            long_or_short: LongOrShort):
        """
        Class constructor.

        :param notional: Notional.
        :param initial_spot: Initial FX spot rate.
        :param strike: FX strike.
        :param domestic_interest_rate: Domestic interest rate.
        :param foreign_interest_rate: Foreign interest rate.
        :param time_to_maturity: Time to maturity.
        :param long_or_short: long_or_short: Indicates if the option is 'LONG' or 'SHORT'.
        """
        self.notional: float = notional
        self.initial_spot: float = initial_spot
        self.strike: float = strike
        self.domestic_interest_rate: float = domestic_interest_rate
        self.foreign_interest_rate: float = foreign_interest_rate
        self.time_to_maturity: float = time_to_maturity

        if long_or_short not in [LongOrShort.LONG, LongOrShort.SHORT]:
            raise ValueError(f'Unknown direction: {long_or_short}')
        else:
            self.long_or_short: LongOrShort = long_or_short

    def get_analytical_price(self) -> float:
        """
        Returns the analytical (discounted) price of the FX Forward.
        """

        drift: float = self.domestic_interest_rate - self.foreign_interest_rate
        direction: float = 1 if self.long_or_short == LongOrShort.LONG else -1

        payoff: float = \
            direction * self.notional * (self.initial_spot * np.exp(drift * self.time_to_maturity) - self.strike)
        discounted_payoff: float = payoff * np.exp(-1 * self.domestic_interest_rate * self.time_to_maturity)
        return discounted_payoff

    def get_time_independent_monte_carlo_price(
            self,
            number_of_paths: int,
            number_of_time_steps: int,
            volatility: float,
            plot_paths: bool = True,
            show_stats: bool = True) -> [MonteCarloPricingResults | str]:
        """
        Returns the price (in domestic currency) for an FX (Foreign Exchange) forward using time-independent GBM
        simulations.

        :param show_stats: Displays the mean, standard deviation, 95% PFE and normality test.
        :param volatility: Time independent volatility.
        :param number_of_paths: Number of Monte Carlo simulation paths.
        :param number_of_time_steps: Number of time steps for the Monte Carlo simulation.
        :param plot_paths: If set to True plots the current_value.
        :return: Monte Carlo price for an FX forward in the domestic currency.
        """
        drift: float = self.domestic_interest_rate - self.foreign_interest_rate
        direction: float = 1 if self.long_or_short == LongOrShort.LONG else -1

        gbm: TimeIndependentGBM = TimeIndependentGBM(drift, volatility, self.initial_spot)
        paths: np.ndarray = \
            gbm.get_paths(
                number_of_paths=number_of_paths,
                number_of_time_steps=number_of_time_steps,
                time_to_maturity=self.time_to_maturity)

        paths = paths * self.notional

        if plot_paths:
            gbm.create_plots(paths, self.time_to_maturity)

        if show_stats:
            gbm.get_path_statistics(paths, self.time_to_maturity)

        payoffs = \
            (paths[:, -1] - self.notional * self.strike) * \
            np.exp(-1 * self.domestic_interest_rate * self.time_to_maturity)

        price: float = direction * np.average(payoffs)
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloPricingResults(price, error)

    def get_time_dependent_monte_carlo_price(
            self,
            number_of_paths: int,
            number_of_time_steps: int,
            volatility_excel_path: str,
            volatility_excel_sheet_name: str,
            plot_paths: bool = False,
            show_stats: bool = False):
        """
        Returns the price (in domestic currency) for an FX (Foreign Exchange) forward using time-dependent GBM
        simulations.

        :param number_of_paths: Number of Monte Carlo simulation paths.
        :param number_of_time_steps: Number of time steps in the Monte Carlo simulation
        :param volatility_excel_path: Specifies the path where the Excel file of volatilities is stored.
        :param volatility_excel_sheet_name: Specifies the name of the Excel sheet where the volatilities are stored.
        :param plot_paths: If set to True plots the current_value.
        :param show_stats: Displays the mean, standard deviation, 95% PFE and normality test.
        :return: Monte Carlo price for an FX forward in the domestic currency.
        """
        drift: float = self.domestic_interest_rate - self.foreign_interest_rate
        direction: float = 1 if self.long_or_short == LongOrShort.LONG else -1

        gbm: TimeDependentGBM = \
            TimeDependentGBM(
                drift=drift,
                excel_file_path=volatility_excel_path,
                sheet_name=volatility_excel_sheet_name,
                initial_spot=self.initial_spot)

        paths: np.ndarray = \
            gbm.get_paths(
                number_of_paths=number_of_paths,
                number_of_time_steps=number_of_time_steps,
                time_to_maturity=self.time_to_maturity)

        paths = paths * self.notional

        if plot_paths:
            gbm.create_plots(paths, self.time_to_maturity)

        if show_stats:
            gbm.get_path_statistics(paths, self.time_to_maturity)

        payoffs = \
            (paths[:, -1] - self.notional * self.strike) * \
            np.exp(-1 * self.domestic_interest_rate * self.time_to_maturity)

        price: float = direction * np.average(payoffs)
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloPricingResults(price, error)
