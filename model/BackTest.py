import pandas as pd
from typing import Union, List
from enum import Enum


class OrderType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


class RuleBasedOrderExecution:
    """
    Name space containing logic of rule based
    trade execution
    """
    @staticmethod
    def higher_lower_flow(y_current_price, y_prediction_price_ahead):
        if y_prediction_price_ahead > y_current_price:
            return OrderType.BUY.value
        elif y_prediction_price_ahead < y_current_price:
            return OrderType.SELL.value
        else:
            return OrderType.HOLD.value

    @staticmethod
    def higher_lower_flow_interval(y_current_price, y_prediction_price_ahead, interval):
        pass


class RuleBasedAmountAllocation:
    """
    Name space containing logic of rule based
    amount allocation per trade
    """
    @staticmethod
    def allocate_const(val: float):
        return val

    @staticmethod
    def kelly_criterion():
        raise NotImplemented


class Balance:
    def __init__(self,
                 init_amount: float):
        self._pnl = []
        self._balance = [init_amount]

    def update(self, val: float):
        self._pnl.append(val)
        self._balance.append(self._balance[-1] + val)

    @property
    def pnl_history(self):
        return self._pnl

    @property
    def balance_history(self):
        return self._balance

    @property
    def balance_current(self):
        return self._balance[-1]


class BackTest:
    def __init__(self,
                 balance: Balance,
                 current_price: pd.Series,
                 prediction_price_ahead: pd.Series,
                 rule_based_trade_order: callable,
                 rule_based_amount_alloc: Union[callable, float],
                 min_trade_interval: int = 13,
                 bid_ask_spread: Union[float, callable] = 0,
                 commision: Union[float, callable] = 0):
        """
        :param current_price: vector of historical prices
        :param prediction_price_ahead: vector of predicted prices ahead of current prices by min_trade_interval into future
        :param rule_based_trade_order: the logic which is used to execute trade order
        :param rule_based_amount_value: the logic which is used to allocate trade amount, could be const or based
        on signal strength or utility function
        :param min_trade_interval: minimum trade price tick interval for order
        :param bid_ask_spread: bid ask spread in %, could be stochastic value sampled from some dist
        :param commision: commision per trade
        """
        self.balance = balance
        self.current_price = current_price
        self.prediction_price_ahead = prediction_price_ahead
        self.rule_based_trade_order = rule_based_trade_order
        self.rule_based_amount_alloc = rule_based_amount_alloc
        self.min_trade_interval = min_trade_interval
        self.bid_ask_spread = bid_ask_spread
        self.commision = commision

    def create_order_flow(self) -> List:
        order_flow = []
        for i, (y_current_price, y_pred_price_ahead) in enumerate(zip(self.current_price, self.prediction_price_ahead)):
            if i % self.min_trade_interval == 0:
                order_flow.append(self.rule_based_trade_order(y_current_price, y_pred_price_ahead))
            else:
                order_flow.append(OrderType.HOLD.value)
        return order_flow

    # TODO: current implementation only supports const val allocation
    def calc_balance(self, order_flow: List) -> None:
        for curr_index_price, (order_type, y_current_price) in enumerate(zip(order_flow, self.current_price)):
            alloc_price = y_current_price + self.bid_ask_spread
            if self.balance.balance_current > 0:
                alloc_amount = self.rule_based_amount_alloc * self.balance.balance_current / alloc_price
            else:
                alloc_amount = 0
            try:
                price_ahead = self.current_price[curr_index_price + self.min_trade_interval]
            except KeyError:
                break
            # assert self.current_price[curr_index_price] == alloc_price, "Oops, price mistmatch"
            if order_type == 1:
                self.balance.update((price_ahead - alloc_price) * alloc_amount - self.commision)
            elif order_type == -1:
                self.balance.update((alloc_price - price_ahead) * alloc_amount - self.commision)
            else:
                self.balance.update(0)


class PortfolioStats:
    def __init__(self, balance: Balance):
        self.balance_history = balance.balance_history
        self.pnl_history = balance.pnl_history
#         Need number of trades on average








