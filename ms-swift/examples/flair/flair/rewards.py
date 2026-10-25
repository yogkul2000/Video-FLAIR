from swift.rewards import ORM

from .scorer import get_scorer


class _FlairComponent(ORM):
    component = None

    def __call__(self, completions, **kwargs):
        return get_scorer().components(completions, **kwargs)[self.component]


class FlairAnswer(_FlairComponent):
    component = 'ans'


class FlairFormat(_FlairComponent):
    component = 'format'


class FlairComply(_FlairComponent):
    component = 'comply'


class FlairCost(_FlairComponent):
    component = 'cost'


class FlairBalance(_FlairComponent):
    component = 'balance'


class FlairSelect(_FlairComponent):
    component = 'select'


class FlairVerifier(_FlairComponent):
    component = 'verifier'


REWARDS = {
    'flair_ans': FlairAnswer,
    'flair_format': FlairFormat,
    'flair_comply': FlairComply,
    'flair_cost': FlairCost,
    'flair_balance': FlairBalance,
    'flair_select': FlairSelect,
    'flair_verifier': FlairVerifier,
}
