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
                 'members': {'available_operations': {'kind': 'method',
                                                      'return_annotation': 'list[PricingCapability]',
                                                      'signature': {'keyword_only': [],
                                                                    'keyword_only_defaults': {},
                                                                    'positional': ['self'],
                                                                    'positional_defaults': {},
                                                                    'required_keyword_only': []}},
                             'source_family': {'kind': 'property',
                                               'return_annotation': "Literal['standard_single_asset', 'stochastic_volatility', 'local_stochastic_volatility', 'rough_family_lsv', 'single_asset_hull_white', 'direct_rough_volatility', 'multi_asset', 'stochastic_dividend', 'stochastic_dividend_lsv']",
                                               'signature': {'keyword_only': [],
                                                             'keyword_only_defaults': {},
                                                             'positional': ['self'],
                                                             'positional_defaults': {},
                                                             'required_keyword_only': []}},
                             'capability': {'kind': 'method',
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
                                                                          'RoughFamilyLsvMeasureRisk | '
                                                                          'RoughVolatilityMeasureRisk | '
                                                                          'StochasticDividendMeasureRisk | '
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

CLASSES['CalibrationMetric'] = {"bases": [], "members": {
    'name': {"kind": "property", "declaration": '@property\ndef name(self) -> str: ...'},
    'values': {"kind": "property", "declaration": '@property\ndef values(self) -> list[float]: ...'},
    'unit': {"kind": "property", "declaration": '@property\ndef unit(self) -> Literal["dimensionless", "count", "indicator", "time", "variance", "equity_coordinate", "brownian_coordinate"]: ...'},
    'convention': {"kind": "property", "declaration": '@property\ndef convention(self) -> str: ...'},
}}

CLASSES['ReportedCalibration'] = {"bases": [], "members": {
    'kind': {"kind": "property", "declaration": '@property\ndef kind(self) -> Literal["particle_lsv", "hull_white_lsv", "local_correlation", "bass_fixed_point", "bass_surface_projection"]: ...'},
    'underlying_ids': {"kind": "property", "declaration": '@property\ndef underlying_ids(self) -> list[int]: ...'},
    'seed': {"kind": "property", "declaration": '@property\ndef seed(self) -> int | None: ...'},
    'axes': {"kind": "property", "declaration": '@property\ndef axes(self) -> list[tuple[str, list[float]]]: ...'},
    'metrics': {"kind": "property", "declaration": '@property\ndef metrics(self) -> list[CalibrationMetric]: ...'},
}}

CLASSES["ValuationReport"]["members"]["calibration_diagnostics"] = {
    "kind": "property",
    "declaration": "@property\ndef calibration_diagnostics(self) -> list[ReportedCalibration] | None: ...",
}

# Measures (ADR 0043, phase 1).
CLASSES['MeasureRequest'] = {"bases": [], "members": {
    'price': {"kind": 'staticmethod', "declaration": '@staticmethod\ndef price() -> MeasureRequest: ...'},
    'spot_delta': {"kind": 'staticmethod', "declaration": '@staticmethod\ndef spot_delta() -> MeasureRequest: ...'},
    'spot_gamma': {"kind": 'staticmethod', "declaration": '@staticmethod\ndef spot_gamma(*, relative_bump: float | None = None, absolute_bump: float | None = None) -> MeasureRequest: ...'},
    'parallel_market_iv_vega': {"kind": 'staticmethod', "declaration": '@staticmethod\ndef parallel_market_iv_vega() -> MeasureRequest: ...'},
    'vega_kt': {"kind": 'staticmethod', "declaration": '@staticmethod\ndef vega_kt(*, maturity_nodes: Sequence[DateLike], log_forward_moneyness_nodes: Sequence[float], relative_density_threshold: float, full_bucket_covariance: bool = False) -> MeasureRequest: ...'},
    'measure': {"kind": 'property', "declaration": '@property\ndef measure(self) -> Literal["price", "spot_delta", "spot_gamma", "parallel_market_iv_vega", "vega_kt"]: ...'},
    'method': {"kind": 'property', "declaration": '@property\ndef method(self) -> Literal["native", "native_spot_bump"]: ...'},
    'relative_bump': {"kind": 'property', "declaration": '@property\ndef relative_bump(self) -> float | None: ...'},
    'absolute_bump': {"kind": 'property', "declaration": '@property\ndef absolute_bump(self) -> float | None: ...'},
    '__eq__': {"kind": 'method', "declaration": 'def __eq__(self, value: object, /) -> bool: ...'},
    '__repr__': {"kind": 'method', "declaration": 'def __repr__(self) -> str: ...'},
}}
CLASSES['MeasureCapability'] = {"bases": [], "members": {
    'request': {"kind": 'property', "declaration": '@property\ndef request(self) -> MeasureRequest: ...'},
    'measure': {"kind": 'property', "declaration": '@property\ndef measure(self) -> Literal["price", "spot_delta", "spot_gamma", "parallel_market_iv_vega", "vega_kt"]: ...'},
    'method': {"kind": 'property', "declaration": '@property\ndef method(self) -> Literal["native", "native_spot_bump"]: ...'},
    'status': {"kind": 'property', "declaration": '@property\ndef status(self) -> str: ...'},
    'reason': {"kind": 'property', "declaration": '@property\ndef reason(self) -> str | None: ...'},
    'detail': {"kind": 'property', "declaration": '@property\ndef detail(self) -> str: ...'},
}}
CLASSES['PricingTask']["members"]['from_source'] = {"kind": 'staticmethod', "declaration": '@staticmethod\ndef from_source(source: PricingSpecification | MultiAssetSpecification | StochasticDividendSpecification, *, operation: Literal["evaluate", "aad", "local_variance_risk", "measures"] = "evaluate", gamma_relative_bump: float | None = None, measures: Sequence[MeasureRequest] | None = None) -> PricingTask: ...'}
CLASSES['PricingTask']["members"]['from_specification'] = {"kind": 'staticmethod', "declaration": '@staticmethod\ndef from_specification(specification: PricingSpecification, *, operation: str = "evaluate", gamma_relative_bump: float | None = None, measures: Sequence[MeasureRequest] | None = None) -> PricingTask: ...'}
CLASSES['PricingTask']["members"]['with_operation'] = {"kind": 'method', "declaration": 'def with_operation(self, operation: str, *, gamma_relative_bump: float | None = None, measures: Sequence[MeasureRequest] | None = None) -> PricingTask: ...'}
CLASSES['PricingTask']["members"]['capability'] = {"kind": 'method', "declaration": 'def capability(self, operation: str, *, gamma_relative_bump: float | None = None, measures: Sequence[MeasureRequest] | None = None) -> PricingCapability: ...'}
CLASSES['PricingTask']["members"]['measures'] = {"kind": 'property', "declaration": '@property\ndef measures(self) -> list[MeasureRequest] | None: ...'}
CLASSES['PricingTask']["members"]['available_measures'] = {"kind": 'method', "declaration": 'def available_measures(self) -> dict[str, list[str]]: ...'}
CLASSES['PricingTask']["members"]['measure_capabilities'] = {"kind": 'method', "declaration": 'def measure_capabilities(self, measures: Sequence[MeasureRequest]) -> list[MeasureCapability]: ...'}
CLASSES['CompiledPricingTask']["members"]['with_operation'] = {"kind": 'method', "declaration": 'def with_operation(self, operation: Literal["evaluate", "aad", "local_variance_risk", "measures"], *, gamma_relative_bump: float | None = None, measures: Sequence[MeasureRequest] | None = None) -> CompiledPricingTask: ...'}
CLASSES['CompiledPricingTask']["members"]['capability_for'] = {"kind": 'method', "declaration": 'def capability_for(self, operation: Literal["evaluate", "aad", "local_variance_risk", "measures"], *, gamma_relative_bump: float | None = None, measures: Sequence[MeasureRequest] | None = None) -> PricingCapability: ...'}
CLASSES['CompiledPricingTask']["members"]['measure_capabilities'] = {"kind": 'method', "declaration": 'def measure_capabilities(self, measures: Sequence[MeasureRequest]) -> list[MeasureCapability]: ...'}
CLASSES['ReportedRisk']["members"]['measure'] = {"kind": 'property', "declaration": '@property\ndef measure(self) -> Literal["price", "spot_delta", "spot_gamma", "parallel_market_iv_vega", "vega_kt"] | None: ...'}
CLASSES['ReportedRisk']["members"]['method_detail'] = {"kind": 'property', "declaration": '@property\ndef method_detail(self) -> dict[str, str | float] | None: ...'}
CLASSES['ValuationReport']["members"]['risk'] = {"kind": 'method', "declaration": 'def risk(self, measure: str) -> ReportedRisk | None: ...'}
CLASSES['ValuationReport']["members"]['to_records'] = {"kind": 'method', "declaration": 'def to_records(self) -> list[dict[str, object]]: ...'}

# Bump-and-revalue (ADR 0043, phase 2).
CLASSES['MeasureRequest']["members"]['bump_and_revalue'] = {"kind": 'staticmethod', "declaration": '@staticmethod\ndef bump_and_revalue(measure: Literal["spot_delta", "spot_gamma", "parallel_market_iv_vega"], *, calibration: Literal["not_applicable", "fixed", "recalibrated"], relative: float | None = None, absolute: float | None = None) -> MeasureRequest: ...'}
CLASSES['MeasureRequest']["members"]['method'] = {"kind": 'property', "declaration": '@property\ndef method(self) -> Literal["native", "native_spot_bump", "bump_and_revalue"]: ...'}
CLASSES['MeasureRequest']["members"]['calibration'] = {"kind": 'property', "declaration": '@property\ndef calibration(self) -> Literal["not_applicable", "fixed", "recalibrated"] | None: ...'}
CLASSES['MeasureCapability']["members"]['method'] = {"kind": 'property', "declaration": '@property\ndef method(self) -> Literal["native", "native_spot_bump", "bump_and_revalue"]: ...'}
CLASSES['PricingTask']["members"]['bump_cost'] = {"kind": 'method', "declaration": 'def bump_cost(self, measures: Sequence[MeasureRequest]) -> dict[str, int]: ...'}
CLASSES['CompiledPricingTask']["members"]['bump_cost'] = {"kind": 'method', "declaration": 'def bump_cost(self) -> dict[str, int]: ...'}

# Calibrated native methods for rough-family LSV (ADR 0043, phase 3 step B1).
CLASSES['MeasureRequest']["members"]['spot_delta'] = {"kind": 'staticmethod', "declaration": '@staticmethod\ndef spot_delta(*, calibration: Literal["not_applicable", "fixed", "recalibrated"] | None = None) -> MeasureRequest: ...'}
CLASSES['MeasureRequest']["members"]['spot_gamma'] = {"kind": 'staticmethod', "declaration": '@staticmethod\ndef spot_gamma(*, relative_bump: float | None = None, absolute_bump: float | None = None, calibration: Literal["not_applicable", "fixed", "recalibrated"] | None = None) -> MeasureRequest: ...'}
CLASSES['MeasureRequest']["members"]['method'] = {"kind": 'property', "declaration": '@property\ndef method(self) -> Literal["native", "native_spot_bump", "native_calibrated", "native_spot_bump_calibrated", "bump_and_revalue"]: ...'}
CLASSES['MeasureCapability']["members"]['method'] = {"kind": 'property', "declaration": '@property\ndef method(self) -> Literal["native", "native_spot_bump", "native_calibrated", "native_spot_bump_calibrated", "bump_and_revalue"]: ...'}

# Native market-IV Vega for rough-family LSV (ADR 0043, phase 3 step B2).
CLASSES['MeasureRequest']["members"]['parallel_market_iv_vega'] = {"kind": 'staticmethod', "declaration": '@staticmethod\ndef parallel_market_iv_vega(*, calibration: Literal["not_applicable", "fixed", "recalibrated"] | None = None) -> MeasureRequest: ...'}

# Model-parameter risk for rough-family LSV (ADR 0043, phase 3 step B3).
_MEASURE_NAMES = '"price", "spot_delta", "spot_gamma", "parallel_market_iv_vega", "vega_kt", "model_parameters"'
CLASSES['MeasureRequest']["members"]['model_parameters'] = {"kind": 'staticmethod', "declaration": '@staticmethod\ndef model_parameters(parameters: Sequence[str], *, calibration: Literal["not_applicable", "fixed", "recalibrated"]) -> MeasureRequest: ...'}
CLASSES['MeasureRequest']["members"]['parameters'] = {"kind": 'property', "declaration": '@property\ndef parameters(self) -> list[str] | None: ...'}
CLASSES['MeasureRequest']["members"]['measure'] = {"kind": 'property', "declaration": f'@property\ndef measure(self) -> Literal[{_MEASURE_NAMES}]: ...'}
CLASSES['MeasureCapability']["members"]['measure'] = {"kind": 'property', "declaration": f'@property\ndef measure(self) -> Literal[{_MEASURE_NAMES}]: ...'}
CLASSES['ReportedRisk']["members"]['measure'] = {"kind": 'property', "declaration": f'@property\ndef measure(self) -> Literal[{_MEASURE_NAMES}] | None: ...'}
CLASSES['ReportedRisk']["members"]['model_parameter'] = {"kind": 'property', "declaration": '@property\ndef model_parameter(self) -> str | None: ...'}
CLASSES['ValuationReport']["members"]['model_parameter_risk'] = {"kind": 'method', "declaration": 'def model_parameter_risk(self, parameter: str) -> ReportedRisk | None: ...'}

# Fixed-model native measures for direct rough volatility (ADR 0043, phase 3
# step C): the calibration treatment of model_parameters becomes optional.
CLASSES['MeasureRequest']["members"]['model_parameters'] = {"kind": 'staticmethod', "declaration": '@staticmethod\ndef model_parameters(parameters: Sequence[str], *, calibration: Literal["not_applicable", "fixed", "recalibrated"] | None = None) -> MeasureRequest: ...'}

# Correlation risk for stochastic dividends (ADR 0043, phase 3 step D2).
_MEASURE_NAMES = '"price", "spot_delta", "spot_gamma", "parallel_market_iv_vega", "vega_kt", "model_parameters", "correlations"'
CLASSES['MeasureRequest']["members"]['correlations'] = {"kind": 'staticmethod', "declaration": '@staticmethod\ndef correlations(correlations: Sequence[str]) -> MeasureRequest: ...'}
CLASSES['MeasureRequest']["members"]['correlation_labels'] = {"kind": 'property', "declaration": '@property\ndef correlation_labels(self) -> list[str] | None: ...'}
CLASSES['MeasureRequest']["members"]['measure'] = {"kind": 'property', "declaration": f'@property\ndef measure(self) -> Literal[{_MEASURE_NAMES}]: ...'}
CLASSES['MeasureCapability']["members"]['measure'] = {"kind": 'property', "declaration": f'@property\ndef measure(self) -> Literal[{_MEASURE_NAMES}]: ...'}
CLASSES['ReportedRisk']["members"]['measure'] = {"kind": 'property', "declaration": f'@property\ndef measure(self) -> Literal[{_MEASURE_NAMES}] | None: ...'}
CLASSES['ReportedRisk']["members"]['correlation'] = {"kind": 'property', "declaration": '@property\ndef correlation(self) -> str | None: ...'}
CLASSES['ValuationReport']["members"]['correlation_risk'] = {"kind": 'method', "declaration": 'def correlation_risk(self, correlation: str) -> ReportedRisk | None: ...'}
