"""Independently frozen Contract API inputs, annotations and outputs."""
def method(declaration): return {'kind':'method','declaration':declaration}
def static(declaration): return {'kind':'staticmethod','declaration':'@staticmethod\n'+declaration}
def prop(declaration): return {'kind':'property','declaration':'@property\n'+declaration}
CLASSES = {
    'Contract': {'bases': [], 'members': {
        'from_json':static('def from_json(json: str) -> Contract: ...'),
        'json_schema':static('def json_schema(version: int = 1) -> str: ...'),
        'bundle':static('def bundle(id: str, contracts: Sequence[Contract]) -> Contract: ...'),
        'to_json':method('def to_json(self) -> str: ...'),
        'fingerprint':prop('def fingerprint(self) -> str: ...'),
        'analysis_json':method('def analysis_json(self) -> str: ...'),
        'observations_json':method('def observations_json(self) -> str: ...'),
        'reversed':method('def reversed(self, id: str) -> Contract: ...'),
        'scaled':method('def scaled(self, id: str, factor: float) -> Contract: ...'),
        'exercise_observation_json':method('def exercise_observation_json(self, option_id: str, exercise_date: DateLike) -> str: ...'),
        'fixing_event_json':method('def fixing_event_json(self, event_id: str, observation_id: str, value: float, *, effective_date: DateLike | None = None) -> str: ...'),
        'advance':method('def advance(self, events_json: str, as_of: DateLike) -> ContractLifecycle: ...'),
        'to_graphs':method('def to_graphs(self, as_of: DateLike, underlying_ids: Sequence[int]) -> list[GraphProduct]: ...'),
        'pricing_task':method('def pricing_task(self, source: PricingSpecification | MultiAssetSpecification | StochasticDividendSpecification, *, operation: Literal["evaluate", "aad", "local_variance_risk"] = "evaluate", gamma_relative_bump: float | None = None, lsm: LsmConfig | None = None, american_grid: Sequence[DateLike] | None = None, brownian_bridge: bool = False) -> PricingTask: ...'),
    }},
    'ContractLifecycle': {'bases': [], 'members': {
        'from_json':static('def from_json(json: str) -> ContractLifecycle: ...'),
        'to_json':method('def to_json(self) -> str: ...'),
        'residual_contract':prop('def residual_contract(self) -> Contract: ...'),
        'as_of':prop('def as_of(self) -> str: ...'),
        'applied_event_ids':prop('def applied_event_ids(self) -> list[str]: ...'),
        'determined_cashflows_json':method('def determined_cashflows_json(self) -> str: ...'),
        'outstanding_payments_json':method('def outstanding_payments_json(self) -> str: ...'),
        'remaining_observations_json':method('def remaining_observations_json(self) -> str: ...'),
        'barrier_states_json':method('def barrier_states_json(self) -> str: ...'),
        'advance':method('def advance(self, events_json: str, as_of: DateLike) -> ContractLifecycle: ...'),
    }},
    'ContractBuilder': {'bases': [], 'members': {
        '__init__':method('def __init__(self, id: str, currency_id: int, underlying_id: int, fixing_source: str) -> None: ...'),
        'bermudan':method('def bermudan(self, side: Literal["call", "put"], strike: float, notional: float, exercise_dates: Sequence[DateLike]) -> Contract: ...'),
        'american':method('def american(self, side: Literal["call", "put"], strike: float, notional: float, start: DateLike, expiry: DateLike) -> Contract: ...'),
        'continuous_barrier':method('def continuous_barrier(self, side: Literal["call", "put"], strike: float, notional: float, expiry: DateLike, payment_date: DateLike, start: DateLike, end: DateLike, direction: Literal["up", "down"], style: Literal["knock_in", "knock_out"], level: float, rebate: float) -> Contract: ...'),
        'forward':method('def forward(self, strike: float, notional: float, observation_date: DateLike, payment_date: DateLike) -> Contract: ...'),
        'european':method('def european(self, side: Literal["call", "put"], strike: float, notional: float, expiry: DateLike, payment_date: DateLike) -> Contract: ...'),
        'digital':method('def digital(self, side: Literal["call", "put"], strike: float, payout: float, expiry: DateLike, payment_date: DateLike) -> Contract: ...'),
        'asian':method('def asian(self, aggregation: Literal["arithmetic_average", "geometric_average", "running_minimum", "running_maximum"], side: Literal["call", "put"], strike: float, notional: float, observation_dates: Sequence[DateLike], payment_date: DateLike) -> Contract: ...'),
        'swap':method('def swap(self, notional: float, fixed_rate: float, observation_dates: Sequence[DateLike], payment_dates: Sequence[DateLike], year_fractions: Sequence[float], *, pay_fixed: bool = True) -> Contract: ...'),
        'discrete_barrier':method('def discrete_barrier(self, side: Literal["call", "put"], strike: float, notional: float, expiry: DateLike, payment_date: DateLike, monitoring_dates: Sequence[DateLike], direction: Literal["up", "down"], style: Literal["knock_in", "knock_out"], level: float, rebate: float) -> Contract: ...'),
        'autocall':method('def autocall(self, notional: float, initial_spot: float, protection_level: float, observation_dates: Sequence[DateLike], payment_dates: Sequence[DateLike], call_levels: Sequence[float], coupon_levels: Sequence[float], coupon_rates: Sequence[float]) -> Contract: ...'),
    }},
}
