"""
Contains a class with collection of static methods for printing results to the console.
"""


class ConsoleUtils:
    """
    A collection of static methods for printing results to the console.
    """
    @staticmethod
    def print_monte_carlo_pricing_results(
            title: str,
            monte_carlo_price: float,
            monte_carlo_price_error: float,
            analytical_price: float) -> None:
        """
        Prints the pricing results of a Monte Carlo simulation.

        :param title: The title of the results.
        :param monte_carlo_price: Monte Carlo price.
        :param monte_carlo_price_error: Error in the Monte Carlo price (standard deviation / sqrt(number of sims)).
        :param analytical_price: Analytical price.
        :return: None.
        """
        print('\n\n')
        print(f' {title}')
        print(f' ' + '-' * len(title))
        print(f'  Monte Carlo Price: {monte_carlo_price:,.2f} ± {monte_carlo_price_error:,.2f}')
        print(f'  Analytical Price: {analytical_price:,.2f}')
