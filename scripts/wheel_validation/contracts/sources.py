"""Independent frozen sources API expectations, migrated from the release gate."""

CLASSES = {'BuehlerDividendModel': {'bases': [],
                          'members': {'__init__': {'declaration': 'def __init__(self, *, mean_reversion: '
                                                                  'float, equity_linkage: float, '
                                                                  'dividend_volatility: float, '
                                                                  'equity_dividend_correlation: float) -> '
                                                                  'None:\n'
                                                                  '    ...',
                                                   'kind': 'method'}}},
 'CompiledPricingSpecification': {'bases': [],
                                  'members': {'evaluate': {'kind': 'method',
                                                           'return_name': 'SpecificationResult',
                                                           'signature': {'keyword_only': [],
                                                                         'keyword_only_defaults': {},
                                                                         'positional': ['self'],
                                                                         'positional_defaults': {},
                                                                         'required_keyword_only': []}},
                                              'plan_fingerprint': {'kind': 'property',
                                                                   'return_name': 'str',
                                                                   'signature': {'keyword_only': [],
                                                                                 'keyword_only_defaults': {},
                                                                                 'positional': ['self'],
                                                                                 'positional_defaults': {},
                                                                                 'required_keyword_only': []}},
                                              'specification_fingerprint': {'kind': 'property',
                                                                            'return_name': 'str',
                                                                            'signature': {'keyword_only': [],
                                                                                          'keyword_only_defaults': {},
                                                                                          'positional': ['self'],
                                                                                          'positional_defaults': {},
                                                                                          'required_keyword_only': []}}}},
 'CompiledPricingTask': {'bases': [],
                         'members': {'task': {'kind': 'property',
                                              'declaration': '@property\ndef task(self) -> PricingTask: ...'},
                                     'with_operation': {'kind': 'method',
                                                        'declaration': 'def with_operation(self, operation: Literal["evaluate", "aad", "local_variance_risk"], *, gamma_relative_bump: float | None = None) -> CompiledPricingTask: ...'},
                                     'capability_for': {'kind': 'method',
                                                        'declaration': 'def capability_for(self, operation: Literal["evaluate", "aad", "local_variance_risk"], *, gamma_relative_bump: float | None = None) -> PricingCapability: ...'},
                                     'capability': {'kind': 'method',
                                                    'return_annotation': 'PricingCapability',
                                                    'signature': {'keyword_only': [],
                                                                  'keyword_only_defaults': {},
                                                                  'positional': ['self'],
                                                                  'positional_defaults': {},
                                                                  'required_keyword_only': []}},
                                     'evaluate': {'kind': 'method',
                                                  'return_annotation': 'PricingTaskResult',
                                                  'signature': {'keyword_only': [],
                                                                'keyword_only_defaults': {},
                                                                'positional': ['self'],
                                                                'positional_defaults': {},
                                                                'required_keyword_only': []}},
                                     'plan_fingerprint': {'kind': 'property',
                                                          'return_annotation': 'str',
                                                          'signature': {'keyword_only': [],
                                                                        'keyword_only_defaults': {},
                                                                        'positional': ['self'],
                                                                        'positional_defaults': {},
                                                                        'required_keyword_only': []}},
                                     'task_fingerprint': {'kind': 'property',
                                                          'return_annotation': 'str',
                                                          'signature': {'keyword_only': [],
                                                                        'keyword_only_defaults': {},
                                                                        'positional': ['self'],
                                                                        'positional_defaults': {},
                                                                        'required_keyword_only': []}}}},
 'EquityModelSpec': {'bases': [],
                     'members': {'local_stochastic_volatility': {'declaration': '@staticmethod\n'
                                                                                'def '
                                                                                'local_stochastic_volatility(target: '
                                                                                'Model, factor: '
                                                                                'VolatilityFactorSpec, '
                                                                                'particles: '
                                                                                'LsvParticleConfig, *, '
                                                                                'forward_density: '
                                                                                'HullWhiteLsvTarget | '
                                                                                'None=None) -> '
                                                                                'EquityModelSpec:\n'
                                                                                '    ...',
                                                                 'kind': 'staticmethod'},
                                 'rough_local_stochastic_volatility': {'declaration': '@staticmethod\n'
                                                                                      'def '
                                                                                      'rough_local_stochastic_volatility(target: '
                                                                                      'Model, model: '
                                                                                      'RoughVolatilityModel, '
                                                                                      'particles: '
                                                                                      'LsvParticleConfig) -> '
                                                                                      'EquityModelSpec:\n'
                                                                                      '    ...',
                                                                       'kind': 'staticmethod'},
                                 'rough_volatility': {'declaration': '@staticmethod\n'
                                                                     'def rough_volatility(model: '
                                                                     'RoughVolatilityModel) -> '
                                                                     'EquityModelSpec:\n'
                                                                     '    ...',
                                                      'kind': 'staticmethod'},
                                 'standard': {'declaration': '@staticmethod\n'
                                                             'def standard(model: Model) -> '
                                                             'EquityModelSpec:\n'
                                                             '    ...',
                                              'kind': 'staticmethod'},
                                 'stochastic_volatility': {'declaration': '@staticmethod\n'
                                                                          'def '
                                                                          'stochastic_volatility(initial_volatility: '
                                                                          'float, factor: '
                                                                          'VolatilityFactorSpec) -> '
                                                                          'EquityModelSpec:\n'
                                                                          '    ...',
                                                           'kind': 'staticmethod'}}},
 'LsvParticleConfig': {'bases': [],
                       'members': {'__init__': {'declaration': 'def __init__(self, *, particle_count: int, '
                                                               'calibration_seed: int, log_bandwidth: float, '
                                                               'minimum_effective_samples: float, '
                                                               'retain_reverse_trace: bool=False) -> None:\n'
                                                               '    ...',
                                                'kind': 'method'}}},
 'MultiAssetHullWhiteConfig': {'bases': [],
                               'members': {'__init__': {'declaration': 'def __init__(self, rate_model: '
                                                                       'HullWhiteModel, rate_correlations: '
                                                                       'Sequence[float], *, lsv_targets: '
                                                                       'Sequence[HullWhiteLsvTarget | None] '
                                                                       '| None=None) -> None:\n'
                                                                       '    ...',
                                                        'kind': 'method'}}},
 'MultiAssetModelSpec': {'bases': [],
                         'members': {'__init__': {'declaration': 'def __init__(self, models: '
                                                                 'Sequence[Model], correlation: '
                                                                 'CorrelationSchedule, *, lsv: '
                                                                 'Sequence[MultiAssetLsvConfig | '
                                                                 'MultiAssetLsv2FactorConfig | '
                                                                 'MultiAssetRoughLsvConfig | None] | '
                                                                 'None=None, driver_correlations: '
                                                                 'Sequence[Sequence[Sequence[float]]] | '
                                                                 'None=None, hull_white: '
                                                                 'MultiAssetHullWhiteConfig | None=None, '
                                                                 'local_correlation: LocalCorrelationConfig '
                                                                 '| None=None) -> None:\n'
                                                                 '    ...',
                                                  'kind': 'method'}}},
 'MultiAssetSpecification': {'bases': [],
                             'members': {'__init__': {'declaration': 'def __init__(self, valuation_date: '
                                                                     'DateLike, product: GraphProduct, '
                                                                     'markets: Sequence[Market], model: '
                                                                     'MultiAssetModelSpec, engine: Engine, '
                                                                     '*, maximum_step: float, '
                                                                     'worker_threads: int, '
                                                                     'reduction_block_size: int | None=None) '
                                                                     '-> None:\n'
                                                                     '    ...',
                                                      'kind': 'method'}}},
 'PricingCapability': {'bases': [],
                       'members': {'detail': {'kind': 'property',
                                              'return_annotation': 'str',
                                              'signature': {'keyword_only': [],
                                                            'keyword_only_defaults': {},
                                                            'positional': ['self'],
                                                            'positional_defaults': {},
                                                            'required_keyword_only': []}},
                                   'gamma_relative_bump': {'kind': 'property',
                                                           'return_annotation': 'float | None',
                                                           'signature': {'keyword_only': [],
                                                                         'keyword_only_defaults': {},
                                                                         'positional': ['self'],
                                                                         'positional_defaults': {},
                                                                         'required_keyword_only': []}},
                                   'operation': {'kind': 'property',
                                                 'return_annotation': 'str',
                                                 'signature': {'keyword_only': [],
                                                               'keyword_only_defaults': {},
                                                               'positional': ['self'],
                                                               'positional_defaults': {},
                                                               'required_keyword_only': []}},
                                   'reason': {'kind': 'property',
                                              'return_annotation': 'str | None',
                                              'signature': {'keyword_only': [],
                                                            'keyword_only_defaults': {},
                                                            'positional': ['self'],
                                                            'positional_defaults': {},
                                                            'required_keyword_only': []}},
                                   'status': {'kind': 'property',
                                              'return_annotation': 'str',
                                              'signature': {'keyword_only': [],
                                                            'keyword_only_defaults': {},
                                                            'positional': ['self'],
                                                            'positional_defaults': {},
                                                            'required_keyword_only': []}}}},
 'PricingModelSpec': {'bases': [],
                      'members': {'__init__': {'declaration': 'def __init__(self, equity: EquityModelSpec, '
                                                              '*, rates: RateModelSpec | None=None) -> '
                                                              'None:\n'
                                                              '    ...',
                                               'kind': 'method'}}},
 'PricingNumerics': {'bases': [],
                     'members': {'__init__': {'declaration': 'def __init__(self, *, worker_threads: int, '
                                                             'reduction_block_size: int | None=None, '
                                                             'maximum_step: float | None=None, lsm: '
                                                             'LsmConfig | None=None) -> None:\n'
                                                             '    ...',
                                              'kind': 'method'}}},
 'PricingSpecification': {'bases': [],
                          'members': {'__init__': {'declaration': 'def __init__(self, valuation_date: '
                                                                  'DateLike, product: Product, market: '
                                                                  'Market, model: PricingModelSpec, engine: '
                                                                  'Engine, risk: RiskRequest, numerics: '
                                                                  'PricingNumerics) -> None:\n'
                                                                  '    ...',
                                                   'kind': 'method'},
                                      'compile': {'kind': 'method',
                                                  'return_name': 'CompiledPricingSpecification',
                                                  'signature': {'keyword_only': [],
                                                                'keyword_only_defaults': {},
                                                                'positional': ['self'],
                                                                'positional_defaults': {},
                                                                'required_keyword_only': []}},
                                      'fingerprint': {'kind': 'property',
                                                      'return_name': 'str',
                                                      'signature': {'keyword_only': [],
                                                                    'keyword_only_defaults': {},
                                                                    'positional': ['self'],
                                                                    'positional_defaults': {},
                                                                    'required_keyword_only': []}},
                                      'from_json': {'kind': 'staticmethod',
                                                    'return_name': 'PricingSpecification',
                                                    'signature': {'keyword_only': [],
                                                                  'keyword_only_defaults': {},
                                                                  'positional': ['json'],
                                                                  'positional_defaults': {},
                                                                  'required_keyword_only': []}},
                                      'from_request': {'kind': 'staticmethod',
                                                       'return_name': 'PricingSpecification',
                                                       'signature': {'keyword_only': ['worker_threads',
                                                                                      'reduction_block_size'],
                                                                     'keyword_only_defaults': {'reduction_block_size': None},
                                                                     'positional': ['request'],
                                                                     'positional_defaults': {},
                                                                     'required_keyword_only': ['worker_threads']}},
                                      'to_json': {'kind': 'method',
                                                  'return_name': 'str',
                                                  'signature': {'keyword_only': [],
                                                                'keyword_only_defaults': {},
                                                                'positional': ['self'],
                                                                'positional_defaults': {},
                                                                'required_keyword_only': []}},
                                      'to_pretty_json': {'kind': 'method',
                                                         'return_name': 'str',
                                                         'signature': {'keyword_only': [],
                                                                       'keyword_only_defaults': {},
                                                                       'positional': ['self'],
                                                                       'positional_defaults': {},
                                                                       'required_keyword_only': []}}}},
 'PricingTask': {'bases': [],
                 'members': {'capability': {'kind': 'method',
                                            'return_annotation': 'PricingCapability',
                                            'signature': {'keyword_only': ['gamma_relative_bump'],
                                                          'keyword_only_defaults': {'gamma_relative_bump': None},
                                                          'positional': ['self', 'operation'],
                                                          'positional_defaults': {},
                                                          'required_keyword_only': []}},
                             'compile': {'kind': 'method',
                                         'return_annotation': 'CompiledPricingTask',
                                         'signature': {'keyword_only': [],
                                                       'keyword_only_defaults': {},
                                                       'positional': ['self'],
                                                       'positional_defaults': {},
                                                       'required_keyword_only': []}},
                             'fingerprint': {'kind': 'property',
                                             'return_annotation': 'str',
                                             'signature': {'keyword_only': [],
                                                           'keyword_only_defaults': {},
                                                           'positional': ['self'],
                                                           'positional_defaults': {},
                                                           'required_keyword_only': []}},
                             'from_json': {'kind': 'staticmethod',
                                           'return_annotation': 'PricingTask',
                                           'signature': {'keyword_only': [],
                                                         'keyword_only_defaults': {},
                                                         'positional': ['json'],
                                                         'positional_defaults': {},
                                                         'required_keyword_only': []}},
                             'from_source': {'declaration': '@staticmethod\n'
                                                            'def from_source(source: PricingSpecification | '
                                                            'MultiAssetSpecification | '
                                                            'StochasticDividendSpecification, *, operation: '
                                                            "Literal['evaluate', 'aad', "
                                                            "'local_variance_risk']='evaluate', "
                                                            'gamma_relative_bump: float | None=None) -> '
                                                            'PricingTask:\n'
                                                            '    ...',
                                             'kind': 'staticmethod'},
                             'from_specification': {'kind': 'staticmethod',
                                                    'return_annotation': 'PricingTask',
                                                    'signature': {'keyword_only': ['operation',
                                                                                   'gamma_relative_bump'],
                                                                  'keyword_only_defaults': {'gamma_relative_bump': None,
                                                                                            'operation': 'evaluate'},
                                                                  'positional': ['specification'],
                                                                  'positional_defaults': {},
                                                                  'required_keyword_only': []}},
                             'to_json': {'kind': 'method',
                                         'return_annotation': 'str',
                                         'signature': {'keyword_only': [],
                                                       'keyword_only_defaults': {},
                                                       'positional': ['self'],
                                                       'positional_defaults': {},
                                                       'required_keyword_only': []}},
                             'to_pretty_json': {'kind': 'method',
                                                'return_annotation': 'str',
                                                'signature': {'keyword_only': [],
                                                              'keyword_only_defaults': {},
                                                              'positional': ['self'],
                                                              'positional_defaults': {},
                                                              'required_keyword_only': []}},
                             'with_operation': {'kind': 'method',
                                                'return_annotation': 'PricingTask',
                                                'signature': {'keyword_only': ['gamma_relative_bump'],
                                                              'keyword_only_defaults': {'gamma_relative_bump': None},
                                                              'positional': ['self', 'operation'],
                                                              'positional_defaults': {},
                                                              'required_keyword_only': []}}}},
 'PricingTaskResult': {'bases': [],
                       'members': {'native_result': {'kind': 'property',
                                                     'return_annotation': 'SpecificationResult | '
                                                                          'MultiAssetPrice | '
                                                                          'StochasticDividendPrice | '
                                                                          'HullWhiteAadRisk | '
                                                                          'LsvLocalVarianceRisk | '
                                                                          'StochasticDividendAadRisk | '
                                                                          'StochasticDividendLocalVarianceRisk',
                                                     'signature': {'keyword_only': [],
                                                                   'keyword_only_defaults': {},
                                                                   'positional': ['self'],
                                                                   'positional_defaults': {},
                                                                   'required_keyword_only': []}},
                                   'report': {'kind': 'property',
                                              'return_annotation': 'ValuationReport',
                                              'signature': {'keyword_only': [],
                                                            'keyword_only_defaults': {},
                                                            'positional': ['self'],
                                                            'positional_defaults': {},
                                                            'required_keyword_only': []}}}},
 'RateModelSpec': {'bases': [],
                   'members': {'deterministic': {'declaration': '@staticmethod\n'
                                                                'def deterministic() -> RateModelSpec:\n'
                                                                '    ...',
                                                 'kind': 'staticmethod'},
                               'hull_white': {'declaration': '@staticmethod\n'
                                                             'def hull_white(model: HullWhiteModel, *, '
                                                             'equity_rate_correlation: float, '
                                                             'vol_rate_correlations: Sequence[float] | '
                                                             'None=None) -> RateModelSpec:\n'
                                                             '    ...',
                                              'kind': 'staticmethod'}}},
 'ReportedRisk': {'bases': [],
                  'members': {'axes': {'kind': 'property',
                                       'return_annotation': 'list[tuple[str, list[float]]]',
                                       'signature': {'keyword_only': [],
                                                     'keyword_only_defaults': {},
                                                     'positional': ['self'],
                                                     'positional_defaults': {},
                                                     'required_keyword_only': []}},
                              'calibration': {'kind': 'property',
                                              'return_annotation': 'str',
                                              'signature': {'keyword_only': [],
                                                            'keyword_only_defaults': {},
                                                            'positional': ['self'],
                                                            'positional_defaults': {},
                                                            'required_keyword_only': []}},
                              'convention': {'kind': 'property',
                                             'return_annotation': 'str',
                                             'signature': {'keyword_only': [],
                                                           'keyword_only_defaults': {},
                                                           'positional': ['self'],
                                                           'positional_defaults': {},
                                                           'required_keyword_only': []}},
                              'method': {'kind': 'property',
                                         'return_annotation': 'str',
                                         'signature': {'keyword_only': [],
                                                       'keyword_only_defaults': {},
                                                       'positional': ['self'],
                                                       'positional_defaults': {},
                                                       'required_keyword_only': []}},
                              'name': {'kind': 'property',
                                       'return_annotation': 'str',
                                       'signature': {'keyword_only': [],
                                                     'keyword_only_defaults': {},
                                                     'positional': ['self'],
                                                     'positional_defaults': {},
                                                     'required_keyword_only': []}},
                              'standard_errors': {'kind': 'property',
                                                  'return_annotation': 'list[float] | None',
                                                  'signature': {'keyword_only': [],
                                                                'keyword_only_defaults': {},
                                                                'positional': ['self'],
                                                                'positional_defaults': {},
                                                                'required_keyword_only': []}},
                              'underlying_ids': {'kind': 'property',
                                                 'return_annotation': 'list[int]',
                                                 'signature': {'keyword_only': [],
                                                               'keyword_only_defaults': {},
                                                               'positional': ['self'],
                                                               'positional_defaults': {},
                                                               'required_keyword_only': []}},
                              'unit': {'kind': 'property',
                                       'return_annotation': 'str',
                                       'signature': {'keyword_only': [],
                                                     'keyword_only_defaults': {},
                                                     'positional': ['self'],
                                                     'positional_defaults': {},
                                                     'required_keyword_only': []}},
                              'values': {'kind': 'property',
                                         'return_annotation': 'list[float]',
                                         'signature': {'keyword_only': [],
                                                       'keyword_only_defaults': {},
                                                       'positional': ['self'],
                                                       'positional_defaults': {},
                                                       'required_keyword_only': []}}}},
 'SpecificationResult': {'bases': [],
                         'members': {'native_result': {'kind': 'property',
                                                       'return_annotation': 'PricingResult | HullWhitePrice '
                                                                            '| LsvPrice',
                                                       'signature': {'keyword_only': [],
                                                                     'keyword_only_defaults': {},
                                                                     'positional': ['self'],
                                                                     'positional_defaults': {},
                                                                     'required_keyword_only': []}},
                                     'specification_fingerprint': {'kind': 'property',
                                                                   'return_name': 'str',
                                                                   'signature': {'keyword_only': [],
                                                                                 'keyword_only_defaults': {},
                                                                                 'positional': ['self'],
                                                                                 'positional_defaults': {},
                                                                                 'required_keyword_only': []}},
                                     'standard_error': {'kind': 'property',
                                                        'return_name': 'float',
                                                        'signature': {'keyword_only': [],
                                                                      'keyword_only_defaults': {},
                                                                      'positional': ['self'],
                                                                      'positional_defaults': {},
                                                                      'required_keyword_only': []}},
                                     'value': {'kind': 'property',
                                               'return_name': 'float',
                                               'signature': {'keyword_only': [],
                                                             'keyword_only_defaults': {},
                                                             'positional': ['self'],
                                                             'positional_defaults': {},
                                                             'required_keyword_only': []}}}},
 'StochasticDividendSpecification': {'bases': [],
                                     'members': {'__init__': {'declaration': 'def __init__(self, equity: '
                                                                             'PricingSpecification, '
                                                                             'dividends: '
                                                                             'BuehlerDividendModel, *, '
                                                                             'maximum_step: float, '
                                                                             'volatility_correlations: '
                                                                             'Sequence[float] | None=None, '
                                                                             'rate_correlation: float | '
                                                                             'None=None) -> None:\n'
                                                                             '    ...',
                                                              'kind': 'method'}}},
 'ValuationReport': {'bases': [],
                     'members': {'calibration_seeds': {'kind': 'property',
                                                       'return_annotation': 'list[int]',
                                                       'signature': {'keyword_only': [],
                                                                     'keyword_only_defaults': {},
                                                                     'positional': ['self'],
                                                                     'positional_defaults': {},
                                                                     'required_keyword_only': []}},
                                 'currency_id': {'kind': 'property',
                                                 'return_annotation': 'int',
                                                 'signature': {'keyword_only': [],
                                                               'keyword_only_defaults': {},
                                                               'positional': ['self'],
                                                               'positional_defaults': {},
                                                               'required_keyword_only': []}},
                                 'estimator': {'kind': 'property',
                                               'return_annotation': 'str',
                                               'signature': {'keyword_only': [],
                                                             'keyword_only_defaults': {},
                                                             'positional': ['self'],
                                                             'positional_defaults': {},
                                                             'required_keyword_only': []}},
                                 'evaluated_paths': {'kind': 'property',
                                                     'return_annotation': 'int',
                                                     'signature': {'keyword_only': [],
                                                                   'keyword_only_defaults': {},
                                                                   'positional': ['self'],
                                                                   'positional_defaults': {},
                                                                   'required_keyword_only': []}},
                                 'from_json': {'kind': 'staticmethod',
                                               'return_annotation': 'ValuationReport',
                                               'signature': {'keyword_only': [],
                                                             'keyword_only_defaults': {},
                                                             'positional': ['json'],
                                                             'positional_defaults': {},
                                                             'required_keyword_only': []}},
                                 'independent_sampling_units': {'kind': 'property',
                                                                'return_annotation': 'int',
                                                                'signature': {'keyword_only': [],
                                                                              'keyword_only_defaults': {},
                                                                              'positional': ['self'],
                                                                              'positional_defaults': {},
                                                                              'required_keyword_only': []}},
                                 'library_version': {'kind': 'property',
                                                     'return_annotation': 'str',
                                                     'signature': {'keyword_only': [],
                                                                   'keyword_only_defaults': {},
                                                                   'positional': ['self'],
                                                                   'positional_defaults': {},
                                                                   'required_keyword_only': []}},
                                 'plan_fingerprint': {'kind': 'property',
                                                      'return_annotation': 'str',
                                                      'signature': {'keyword_only': [],
                                                                    'keyword_only_defaults': {},
                                                                    'positional': ['self'],
                                                                    'positional_defaults': {},
                                                                    'required_keyword_only': []}},
                                 'platform': {'kind': 'property',
                                              'return_annotation': 'str',
                                              'signature': {'keyword_only': [],
                                                            'keyword_only_defaults': {},
                                                            'positional': ['self'],
                                                            'positional_defaults': {},
                                                            'required_keyword_only': []}},
                                 'risks': {'kind': 'property',
                                           'return_annotation': 'list[ReportedRisk]',
                                           'signature': {'keyword_only': [],
                                                         'keyword_only_defaults': {},
                                                         'positional': ['self'],
                                                         'positional_defaults': {},
                                                         'required_keyword_only': []}},
                                 'standard_error': {'kind': 'property',
                                                    'return_annotation': 'float',
                                                    'signature': {'keyword_only': [],
                                                                  'keyword_only_defaults': {},
                                                                  'positional': ['self'],
                                                                  'positional_defaults': {},
                                                                  'required_keyword_only': []}},
                                 'task_fingerprint': {'kind': 'property',
                                                      'return_annotation': 'str',
                                                      'signature': {'keyword_only': [],
                                                                    'keyword_only_defaults': {},
                                                                    'positional': ['self'],
                                                                    'positional_defaults': {},
                                                                    'required_keyword_only': []}},
                                 'to_json': {'kind': 'method',
                                             'return_annotation': 'str',
                                             'signature': {'keyword_only': [],
                                                           'keyword_only_defaults': {},
                                                           'positional': ['self'],
                                                           'positional_defaults': {},
                                                           'required_keyword_only': []}},
                                 'to_pretty_json': {'kind': 'method',
                                                    'return_annotation': 'str',
                                                    'signature': {'keyword_only': [],
                                                                  'keyword_only_defaults': {},
                                                                  'positional': ['self'],
                                                                  'positional_defaults': {},
                                                                  'required_keyword_only': []}},
                                 'uncertainty_scope': {'kind': 'property',
                                                       'return_annotation': 'str',
                                                       'signature': {'keyword_only': [],
                                                                     'keyword_only_defaults': {},
                                                                     'positional': ['self'],
                                                                     'positional_defaults': {},
                                                                     'required_keyword_only': []}},
                                 'value': {'kind': 'property',
                                           'return_annotation': 'float',
                                           'signature': {'keyword_only': [],
                                                         'keyword_only_defaults': {},
                                                         'positional': ['self'],
                                                         'positional_defaults': {},
                                                         'required_keyword_only': []}},
                                 'vega_kt': {'kind': 'property',
                                             'return_annotation': 'VegaKtResult | None',
                                             'signature': {'keyword_only': [],
                                                           'keyword_only_defaults': {},
                                                           'positional': ['self'],
                                                           'positional_defaults': {},
                                                           'required_keyword_only': []}}}},
 'VolatilityFactorSpec': {'bases': [],
                          'members': {'bergomi': {'declaration': '@staticmethod\n'
                                                                 'def bergomi(mean_reversion: float, '
                                                                 'vol_of_vol: float, correlation: float) -> '
                                                                 'VolatilityFactorSpec:\n'
                                                                 '    ...',
                                                  'kind': 'staticmethod'},
                                      'bergomi_two_factor': {'declaration': '@staticmethod\n'
                                                                            'def '
                                                                            'bergomi_two_factor(mean_reversions: '
                                                                            'Sequence[float], vol_of_vol: '
                                                                            'float, mixing_weight: float, '
                                                                            'spot_correlations: '
                                                                            'Sequence[float], '
                                                                            'factor_correlation: float) -> '
                                                                            'VolatilityFactorSpec:\n'
                                                                            '    ...',
                                                             'kind': 'staticmethod'},
                                      'rough_bergomi': {'declaration': '@staticmethod\n'
                                                                       'def rough_bergomi(model: '
                                                                       'RoughBergomiModel) -> '
                                                                       'VolatilityFactorSpec:\n'
                                                                       '    ...',
                                                        'kind': 'staticmethod'}}}}
