"""The accepted coupled matrices through dated ledgers and atomic risk checks."""
import test_vega_kt_custom_basis_provider as named


class CoupledBasisProviderTests(named.CustomBasisProviderTests):
    policies = ('coupled_row_hats',)
