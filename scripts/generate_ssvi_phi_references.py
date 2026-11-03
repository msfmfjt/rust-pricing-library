"""Print independent SSVI phi references using only Python's decimal library.

Binary64 input values are lifted exactly into 100-digit Decimal arithmetic.
The Heston-like oracle integrates the exponential power series term by term,
independently of the production closed form and its small-argument polynomial.
This development helper is not used by Rust tests or at runtime.
"""

from decimal import Decimal, localcontext
import json
import math


def heston_integrals(z: Decimal) -> tuple[Decimal, Decimal, int]:
    """Return integral (1-u)*exp(-z*u) and its derivative over u in [0,1]."""
    exponential_term = Decimal(1)
    value = Decimal(0)
    derivative = Decimal(0)
    for order in range(1000):
        value_term = exponential_term / ((order + 1) * (order + 2))
        derivative_term = -exponential_term / ((order + 2) * (order + 3))
        value += value_term
        derivative += derivative_term
        if order > z and max(abs(value_term), abs(derivative_term)) < Decimal('1e-94'):
            return value, derivative, order + 1
        exponential_term *= -z / (order + 1)
    raise AssertionError('Reference series did not converge')


def generate() -> dict:
    with localcontext() as context:
        context.prec = 100
        heston = []
        for z_float in [
            1e-8, math.nextafter(1e-4, 0.0), 1e-4,
            math.nextafter(1e-4, math.inf), 2e-4, 1e-3, 1e-2, 0.1,
            math.nextafter(0.5, 0.0), 0.5, math.nextafter(0.5, math.inf),
            0.75, 2.0, 16.0,
        ]:
            z = Decimal.from_float(z_float)
            value, derivative, terms = heston_integrals(z)
            exponential = (-z).exp()
            closed_value = (z - 1 + exponential) / z**2
            closed_derivative = (2 - z - (z + 2) * exponential) / z**3
            assert abs(value - closed_value) < Decimal('1e-68')
            assert abs(derivative - closed_derivative) < Decimal('1e-68')
            heston.append({'z': z_float, 'value': str(value),
                           'derivative_by_z': str(derivative), 'terms': terms})
        power = []
        for eta_float, gamma_float, theta_float in [
            (0.75, 0.25, 0.04), (0.375, 0.6, 0.125),
            (1.25, 0.125, 2.0), (0.5, 0.5, 0.25),
            (0.5, 0.5, 4.0), (0.625, 0.875, 16.0),
            (1.5, 0.375, 1e-8), (0.125, 0.25, 32.0),
        ]:
            eta, gamma, theta = map(Decimal.from_float, (eta_float, gamma_float, theta_float))
            value = (eta.ln() - gamma * theta.ln() - (1 - gamma) * (1 + theta).ln()).exp()
            derivative = -eta * (gamma + theta) * (
                -(gamma + 1) * theta.ln() - (2 - gamma) * (1 + theta).ln()).exp()
            power.append({'eta': eta_float, 'gamma': gamma_float, 'theta': theta_float,
                          'value': str(value), 'theta_derivative': str(derivative)})
        return {
            'decimal_digits': context.prec,
            'inputs': 'Decimal.from_float lifts each binary64 input exactly.',
            'heston_definition': 'phi(z)=integral_0^1 (1-u)*exp(-z*u) du; phi_z=-integral_0^1 u*(1-u)*exp(-z*u) du.',
            'heston_crosscheck': '100-digit exponential closed form agrees to absolute 1e-68.',
            'degree_14_truncation_at_z_one_half': {
                'value_bound': str(Decimal('0.5')**15 / math.factorial(17)),
                'derivative_bound': str(15 * Decimal('0.5')**14 / math.factorial(17)),
            },
            'heston': heston, 'power_law': power,
        }


if __name__ == '__main__':
    print(json.dumps(generate(), indent=2))
