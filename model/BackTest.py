import numpy as np
from typing import Union, List
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import tqdm


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
    def higher_lower_flow(y_prediction_return_ahead, *args):
        if y_prediction_return_ahead > 0:
            return OrderType.BUY.value
        elif y_prediction_return_ahead < 0:
            return OrderType.SELL.value
        else:
            return OrderType.HOLD.value

    @staticmethod
    def higher_lower_flow_interval(y_prediction_return_ahead, *args):
        upper_interval, lower_interval = args
        if y_prediction_return_ahead > upper_interval:
            return OrderType.BUY.value
        elif y_prediction_return_ahead < lower_interval:
            return OrderType.SELL.value
        else:
            return OrderType.HOLD.value


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

    @property
    def count_winning_directions(self):
        return np.array(self.pnl_history) > 0



class BackTest:
    def __init__(self,
                 balance: Balance,
                 current_price: np.array,
                 prediction_return_ahead: np.array,
                 rule_based_trade_order: callable,
                 rule_based_amount_alloc: Union[callable, float],
                 min_trade_interval: int = 13,
                 pred_trade_interval: int = 13,
                 bid_ask_spread: Union[float, callable] = 0,
                 commision: Union[float, callable] = 0,
                 *args):
        """
        :param current_price: vector of historical prices
        :param prediction_return_ahead: vector of predicted prices ahead of current prices by min_trade_interval into future
        :param rule_based_trade_order: the logic which is used to execute trade order
        :param rule_based_amount_value: the logic which is used to allocate trade amount, could be const or based
        on signal strength or utility function
        :param min_trade_interval: minimum trade price tick interval for order
        :param bid_ask_spread: bid ask spread in %, could be stochastic value sampled from some dist
        :param commision: commision per trade
        """
        self.balance = balance
        self.current_price = current_price
        self.prediction_return_ahead = prediction_return_ahead
        self.rule_based_trade_order = rule_based_trade_order
        self.rule_based_amount_alloc = rule_based_amount_alloc
        self.min_trade_interval = min_trade_interval
        self.pred_trade_interval = pred_trade_interval
        self.bid_ask_spread = bid_ask_spread
        self.commision = commision
        self.arguments_rules_based_order = args

    def create_order_flow(self) -> List:
        order_flow = []
        for i,  y_pred_return_ahead in enumerate(self.prediction_return_ahead):
            if i % self.min_trade_interval == 0:
                order_flow.append(self.rule_based_trade_order(y_pred_return_ahead, *self.arguments_rules_based_order))
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
                price_ahead = self.current_price[curr_index_price + self.pred_trade_interval]
            except IndexError:
                break
            # assert self.current_price[curr_index_price] == alloc_price, "Oops, price mistmatch"
            if order_type == 1:
                self.balance.update((price_ahead - alloc_price) * alloc_amount - self.commision)
            elif order_type == -1:
                self.balance.update((alloc_price - price_ahead) * alloc_amount - self.commision)
            else:
                self.balance.update(0)


def plot_cumulative_time_per_trade(order_flow, time_index):
    cumulative_time_per_trade = [1]
    start_time = time_index[0]
    for order, cur_time in zip(order_flow, time_index):
        if order != 0:
            cumulative_time_per_trade.append((cur_time - start_time) / len(cumulative_time_per_trade))
    cumulative_time_per_trade = cumulative_time_per_trade[1:]
    total_time_in_seconds = (time_index[-1] - time_index[0])
    number_of_trades = np.sum(pd.Series(order_flow) != 0)
    plt.plot(cumulative_time_per_trade)
    plt.axhline(total_time_in_seconds / number_of_trades, color='r', label="Average")
    plt.title('cumulative time per trade in seconds')
    plt.ylabel('time in seconds')
    plt.legend()


def bootstrap_pnl(pnl_series, boostrap_frac: float, bootstrap_round: int):
    average_pnl = []
    for _ in tqdm.tqdm(range(bootstrap_round)):
        average_pnl.append(np.mean(pnl_series.sample(frac=boostrap_frac, replace=True)))
    t_score, p_val = stats.ttest_1samp(average_pnl, popmean=0)
    avg = np.mean(average_pnl)
    plt.hist(average_pnl, bins=int(bootstrap_round/10))
    plt.title("Bootstrapped Avg PnL")
    plt.axvline(avg, color='r',label = f"avg val {avg}, t_test: {t_score}, p_val: {p_val}")
    plt.legend()
    return average_pnl








