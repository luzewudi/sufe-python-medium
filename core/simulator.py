import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from core.utils.log_kit import logger
class Simulator:
    """
    股票交易模拟器
    基于选股结果进行仓位调整，支持开盘价和收盘价交易
    """
    
    def __init__(self, initial_cash: float, commission_rate: float, stamp_tax_rate: float):
        """
        初始化模拟器
        
        :param initial_cash: 初始资金
        :param commission_rate: 券商佣金费率
        :param stamp_tax_rate: 印花税率
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        
        # 持仓信息：股票代码 -> 持仓数量
        self.positions = {}  # {stock_code: shares}
        
        # 统计信息
        self.total_commission = 0.0
        self.total_stamp_tax = 0.0
        self.trade_records = []  # 交易记录
        
        # 交易所限制
        self.exchange_limits = {
            'SH': 100,    # 上海交易所：100股为1手
            'SZ': 100,    # 深圳交易所：100股为1手
            'BJ': 100,    # 北京交易所：100股为1手
            'HK': 100,    # 港股：100股为1手
        }
        
    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """
        计算当前总资产
        
        :param current_prices: 当前价格字典 {stock_code: price}
        :return: 总资产
        """
        position_value = 0.0
        for stock_code, shares in self.positions.items():
            if stock_code in current_prices and not pd.isna(current_prices[stock_code]):
                position_value += shares * current_prices[stock_code]
        
        return self.cash + position_value
    
    def get_exchange(self, stock_code: str) -> str:
        """
        根据股票代码判断所属交易所
        
        :param stock_code: 股票代码
        :return: 交易所代码
        """
        if stock_code.startswith(('60', '68', '90')):
            return 'SH'  # 上海交易所
        elif stock_code.startswith(('00', '30', '20')):
            return 'SZ'  # 深圳交易所
        elif stock_code.startswith(('43', '83', '87')):
            return 'BJ'  # 北京交易所
        elif stock_code.startswith(('0', '1')):
            return 'HK'  # 港股
        else:
            return 'SH'  # 默认上海交易所
    
    def apply_exchange_limits(self, target_shares: int, stock_code: str) -> int:
        """
        应用交易所限制，确保股数符合最小交易单位
        
        :param target_shares: 目标股数
        :param stock_code: 股票代码
        :return: 调整后的股数
        """
        exchange = self.get_exchange(stock_code)
        min_unit = self.exchange_limits.get(exchange, 100)
        
        if target_shares <= 0:
            return 0
        
        # 向下取整到最小交易单位的倍数
        adjusted_shares = (target_shares // min_unit) * min_unit
        
        if adjusted_shares < min_unit:
            return 0  # 如果不足最小交易单位，则不交易
            
        return adjusted_shares
    
    def calculate_target_positions(self, target_ratios: Dict[str, float], 
                                 current_prices: Dict[str, float]) -> Dict[str, int]:
        """
        计算目标持仓数量
        
        :param target_ratios: 目标资金占比 {stock_code: ratio}
        :param current_prices: 当前价格 {stock_code: price}
        :return: 目标持仓数量 {stock_code: shares}
        """
        total_equity = self.get_total_equity(current_prices)
        target_positions = {}
        
        for stock_code, ratio in target_ratios.items():
            if ratio <= 0 or stock_code not in current_prices or pd.isna(current_prices[stock_code]):
                target_positions[stock_code] = 0
                continue
                
            target_value = total_equity * ratio
            price = current_prices[stock_code]
            
            # 计算目标股数
            target_shares = int(target_value / price)
            
            # 应用交易所限制
            target_shares = self.apply_exchange_limits(target_shares, stock_code)
                
            target_positions[stock_code] = target_shares
            
        return target_positions
    
    def adjust_positions(self, target_positions: Dict[str, int], 
                        current_prices: Dict[str, float], 
                        trade_date: str) -> Tuple[float, float]:
        """
        调整仓位到目标持仓
        
        :param target_positions: 目标持仓 {stock_code: shares}
        :param current_prices: 当前价格 {stock_code: price}
        :param trade_date: 交易日期
        :return: (佣金, 印花税)
        """
        commission = 0.0
        stamp_tax = 0.0
        
        # 获取所有涉及的股票代码
        all_stocks = set(self.positions.keys()) | set(target_positions.keys())
        
        for stock_code in all_stocks:
            current_shares = self.positions.get(stock_code, 0)
            target_shares = target_positions.get(stock_code, 0)
            
            if current_shares == target_shares:
                continue
                
            if stock_code not in current_prices or pd.isna(current_prices[stock_code]):
                logger.warning(f"股票 {stock_code} 价格无效，跳过交易")
                continue
                
            price = current_prices[stock_code]
            shares_diff = target_shares - current_shares
            
            if shares_diff > 0:  # 买入
                trade_value = shares_diff * price
                trade_commission = trade_value * self.commission_rate
                commission += trade_commission
                
                # 检查资金是否足够
                required_cash = trade_value + trade_commission
                if required_cash > self.cash:
                    # 资金不足，按比例减少买入数量
                    available_cash = self.cash - trade_commission
                    if available_cash > 0:
                        actual_shares = int(available_cash / price)
                        actual_trade_value = actual_shares * price
                        actual_commission = actual_trade_value * self.commission_rate
                        commission = commission - trade_commission + actual_commission
                        shares_diff = actual_shares
                        trade_value = actual_trade_value
                    else:
                        shares_diff = 0
                        trade_value = 0
                        commission -= trade_commission
                
                if shares_diff > 0:
                    self.cash -= trade_value + trade_commission
                    self.positions[stock_code] = current_shares + shares_diff
                    
                    # 记录交易
                    self.trade_records.append({
                        'date': trade_date,
                        'stock_code': stock_code,
                        'action': '买入',
                        'shares': shares_diff,
                        'price': price,
                        'value': trade_value,
                        'commission': trade_commission,
                        'stamp_tax': 0.0
                    })
                    
            elif shares_diff < 0:  # 卖出
                trade_value = abs(shares_diff) * price
                trade_commission = trade_value * self.commission_rate
                trade_stamp_tax = trade_value * self.stamp_tax_rate
                
                commission += trade_commission
                stamp_tax += trade_stamp_tax
                
                self.cash += trade_value - trade_commission - trade_stamp_tax
                self.positions[stock_code] = current_shares + shares_diff
                
                # 如果持仓为0，删除记录
                if self.positions[stock_code] == 0:
                    del self.positions[stock_code]
                
                # 记录交易
                self.trade_records.append({
                    'date': trade_date,
                    'stock_code': stock_code,
                    'action': '卖出',
                    'shares': abs(shares_diff),
                    'price': price,
                    'value': trade_value,
                    'commission': trade_commission,
                    'stamp_tax': trade_stamp_tax
                })
        
        self.total_commission += commission
        self.total_stamp_tax += stamp_tax
        
        return commission, stamp_tax
    
    def get_position_values(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        获取当前持仓价值
        
        :param current_prices: 当前价格 {stock_code: price}
        :return: 持仓价值 {stock_code: value}
        """
        position_values = {}
        for stock_code, shares in self.positions.items():
            if stock_code in current_prices and not pd.isna(current_prices[stock_code]):
                position_values[stock_code] = shares * current_prices[stock_code]
            else:
                position_values[stock_code] = 0.0
        return position_values
    
    def get_summary(self) -> Dict:
        """
        获取模拟器摘要信息
        
        :return: 摘要信息字典
        """
        return {
            'initial_cash': self.initial_cash,
            'current_cash': self.cash,
            'total_commission': self.total_commission,
            'total_stamp_tax': self.total_stamp_tax,
            'total_trades': len(self.trade_records),
            'current_positions': len(self.positions)
        }
