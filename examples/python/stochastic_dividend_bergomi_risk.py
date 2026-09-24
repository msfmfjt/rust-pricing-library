"""Bergomi model-parameter risk at fixed correlations and time grid."""
# Reuse the fully specified request and plan from the first-order example.
from stochastic_dividend_risk import plan

risk = plan.evaluate_bergomi_aad()
print('Method:', risk.method)
for label, derivative, se in zip(risk.parameter_labels, risk.derivatives, risk.standard_errors):
    if label.startswith('bergomi_'):
        print(f'{label}: {derivative:.8g} (sampling SE {se:.3g})')
