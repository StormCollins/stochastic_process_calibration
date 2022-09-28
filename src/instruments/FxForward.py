import numpy as np
from scipy.stats import norm
from src.monte_carlo_result import MonteCarloResult
from src.gbm.time_independent_gbm import TimeIndependentGBM


class FxForward:
    """
    A class for an FX (Foreign Exchange) Forward.
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
            time_to_maturity: float):
        """
        FX Forward constructor.

        :param notional: Notional.
        :param initial_spot: Initial FX spot rate.
        :param strike: FX strike.
        :param domestic_interest_rate: Domestic interest rate.
        :param foreign_interest_rate: Foreign interest rate.
        :param time_to_maturity: Time to maturity.
        """
        self.notional: float = notional
        self.initial_spot: float = initial_spot
        self.strike: float = strike
        self.domestic_interest_rate: float = domestic_interest_rate
        self.foreign_interest_rate: float = foreign_interest_rate
        self.time_to_maturity: float = time_to_maturity

    def get_analytical_price(self) -> float:
        """
        Returns the analytical (discounted) FX forward price.
        """
        payoff: float = \
            self.notional * self.initial_spot * \
            (np.exp((self.domestic_interest_rate - self.foreign_interest_rate) * self.time_to_maturity) - self.strike)

        discounted_payoff: float = payoff * np.exp(-1 * self.domestic_interest_rate * self.time_to_maturity)
        return discounted_payoff

    def get_time_independent_monte_carlo_price(
            self,
            number_of_paths: int,
            number_of_time_steps: int,
            volatility: float,
            plot_paths: bool = True,
            show_stats: bool = True) -> [MonteCarloResult | str]:
        """
        Returns the price (in domestic currency) for an FX (Foreign Exchange) forward using time independent GBM
        simulations.

        :param show_stats: Displays the mean, standard deviation, 95% PFE and normality test.
        :param volatility: Time independent volatility.
        :param number_of_paths: Number of Monte Carlo simulation paths.
        :param number_of_time_steps: Number of time steps for the Monte Carlo simulation.
        :param plot_paths: If set to True plots the current_value.
        :return: Monte Carlo price for an FX forward in the domestic currency.
        """
        drift: float = self.domestic_interest_rate - self.foreign_interest_rate
        gbm: TimeIndependentGBM = TimeIndependentGBM(drift, volatility)
        paths: np.ndarray = \
            gbm.get_paths(
                number_of_paths=number_of_paths,
                number_of_time_steps=number_of_time_steps,
                notional=self.notional,
                initial_spot=self.initial_spot,
                time_to_maturity=self.time_to_maturity)

        if plot_paths:
            gbm.create_plots(paths, drift, volatility, self.time_to_maturity)

        # if show_stats:
        #     statistics(paths, initial_spot, drift, volatility, time_to_maturity)

        payoffs = \
            (paths[:, -1] - self.notional * self.strike) * \
            np.exp(-1 * self.domestic_interest_rate * self.time_to_maturity)

        price: float = np.average(payoffs)
        error = norm.ppf(0.95) * np.std(payoffs) / np.sqrt(number_of_paths)
        return MonteCarloResult(price, error)
