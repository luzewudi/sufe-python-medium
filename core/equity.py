import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import itertools
from datetime import timedelta
from core.simulator import Simulator
from core.figure import show_performance_plot, save_performance
from core.utils.log_kit import logger


def load_stock_data(runtime_folder: Path) -> Dict[str, pd.DataFrame]:
    """
    加载股票行情数据
    
    :param runtime_folder: 数据路径
    :return: 股票数据字典
    """
    logger.debug("📊 加载股票行情数据...")
    
    # 加载pivot数据
    pivot_file = runtime_folder / "全部股票行情pivot.pkl"
    if not pivot_file.exists():
        raise FileNotFoundError(f"股票行情数据文件不存在: {pivot_file}")
    
    pivot_dict = pd.read_pickle(pivot_file)
    logger.debug(f"✅ 股票行情数据加载完成，包含 {len(pivot_dict)} 个价格类型")
    
    return pivot_dict


def load_select_results(select_results_path: Path) -> pd.DataFrame:
    """
    加载选股结果
    
    :param select_results_path: 选股结果文件路径
    :return: 选股结果DataFrame
    """
    logger.debug("📈 加载选股结果...")
    
    if not select_results_path.exists():
        raise FileNotFoundError(f"选股结果文件不存在: {select_results_path}")
    
    select_results = pd.read_pickle(select_results_path)
    logger.debug(f"✅ 选股结果加载完成，包含 {len(select_results)} 条记录")
    
    return select_results


def get_trading_dates(start_date: str, end_date: str, data_path: Path) -> pd.DatetimeIndex:
    """
    获取交易日列表
    
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param data_path: 数据路径
    :return: 交易日列表
    """
    # 从指数数据中获取交易日
    index_file = data_path /"stock-main-index-data"/ "sh000300.csv"
    if not index_file.exists():
        raise FileNotFoundError(f"指数数据文件不存在: {index_file}")
    
    index_df = pd.read_csv(index_file)
    index_df['交易日期'] = pd.to_datetime(index_df['candle_end_time'])
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    trading_dates = index_df[
        (index_df['交易日期'] >= start_dt) & 
        (index_df['交易日期'] <= end_dt)
    ]['交易日期'].sort_values().reset_index(drop=True)
    
    return trading_dates


def simulate_trading(select_results: pd.DataFrame, 
                    stock_data: Dict[str, pd.DataFrame],
                    trading_dates: pd.DatetimeIndex,
                    initial_cash: float,
                    commission_rate: float,
                    stamp_tax_rate: float,
                    rebalance_time: str = "open") -> pd.DataFrame:
    """
    模拟交易过程
    
    :param select_results: 选股结果
    :param stock_data: 股票行情数据
    :param trading_dates: 交易日列表
    :param initial_cash: 初始资金
    :param commission_rate: 佣金费率
    :param stamp_tax_rate: 印花税率
    :param rebalance_time: 调仓时间 (open/close)
    :return: 账户资金曲线
    """
    logger.debug("🎯 开始模拟交易...")
    
    # 初始化模拟器
    simulator = Simulator(initial_cash, commission_rate, stamp_tax_rate)
    
    # 准备结果记录
    account_records = []
    
    # 获取所有涉及的股票代码
    all_stocks = sorted(select_results['股票代码'].unique())
    logger.debug(f"📊 涉及股票数量: {len(all_stocks)}")
    
    # 按交易日进行模拟
    for i, trade_date in enumerate(trading_dates):
        trade_date_str = trade_date.strftime('%Y-%m-%d')
        
        # 获取当日的选股结果
        daily_selection = select_results[select_results['选股日期'] == trade_date].copy()
        
        if daily_selection.empty:
            logger.debug(f"⚠️ {trade_date_str} 无选股结果，跳过")
            # 记录无交易日的账户状态
            current_prices = {}
            for stock in all_stocks:
                if stock in stock_data.get('close', pd.DataFrame()).columns:
                    current_prices[stock] = stock_data['close'].loc[trade_date, stock] if trade_date in stock_data['close'].index else np.nan
                else:
                    current_prices[stock] = np.nan
            
            total_equity = simulator.get_total_equity(current_prices)
            position_values = simulator.get_position_values(current_prices)
            
            account_records.append({
                '交易日期': trade_date,
                '账户可用资金': simulator.cash,
                '持仓市值': sum(position_values.values()),
                '总资产': total_equity,
                '净值': total_equity / initial_cash,
                '印花税': 0.0,
                '券商佣金': 0.0,
                '手续费': 0.0,
                '涨跌幅': 0.0
            })
            continue
        
        # 获取下一个交易日的价格数据（选股日期早于交易日期一个交易日）
        if i + 1 < len(trading_dates):
            next_trade_date = trading_dates[i + 1]
            price_date = next_trade_date
            # logger.debug(f"📅 {trade_date_str} 选股，使用 {next_trade_date.strftime('%Y-%m-%d')} 的价格交易")
        else:
            # 如果是最后一个交易日，使用当日价格
            price_date = trade_date
            logger.warning(f"⚠️ {trade_date_str} 是最后一个交易日，使用当日价格")
        
        # 获取价格数据
        price_type = 'open' if rebalance_time == 'open' else 'close'
        if price_type not in stock_data:
            logger.warning(f"⚠️ 价格类型 {price_type} 不存在，使用收盘价")
            price_type = 'close'
        
        current_prices = {}
        for stock in all_stocks:
            if stock in stock_data[price_type].columns and price_date in stock_data[price_type].index:
                current_prices[stock] = stock_data[price_type].loc[price_date, stock]
            else:
                current_prices[stock] = np.nan
        
        # 构建目标资金占比字典
        target_ratios = {}
        for _, row in daily_selection.iterrows():
            stock_code = row['股票代码']
            target_ratio = row['目标资金占比']
            if not pd.isna(target_ratio) and target_ratio > 0:
                target_ratios[stock_code] = target_ratio
        
        # 计算目标持仓
        target_positions = simulator.calculate_target_positions(
            target_ratios, current_prices
        )
        
        # 调整仓位
        commission, stamp_tax = simulator.adjust_positions(
            target_positions, current_prices, trade_date_str
        )
        
        # 计算当前总资产
        total_equity = simulator.get_total_equity(current_prices)
        position_values = simulator.get_position_values(current_prices)
        
        # 计算涨跌幅
        if len(account_records) > 0:
            prev_net_value = account_records[-1]['净值']
            current_net_value = total_equity / initial_cash
            pct_change = (current_net_value - prev_net_value) / prev_net_value if prev_net_value > 0 else 0.0
        else:
            pct_change = 0.0
        
        # 记录账户状态
        account_records.append({
            '交易日期': trade_date,
            '账户可用资金': simulator.cash,
            '持仓市值': sum(position_values.values()),
            '总资产': total_equity,
            '净值': total_equity / initial_cash,
            '印花税': stamp_tax,
            '券商佣金': commission,
            '手续费': commission + stamp_tax,
            '涨跌幅': pct_change
        })
        
        # 每100个交易日输出一次进度
        if len(account_records) % 100 == 0:
            logger.debug(f"📈 已处理 {len(account_records)} 个交易日，当前净值: {total_equity/initial_cash:.4f}")
    
    # 转换为DataFrame
    account_df = pd.DataFrame(account_records)
    
    # 添加杠杆信息
    account_df['杠杆'] = 1.0
    account_df['实际杠杆'] = account_df['持仓市值'] / account_df['总资产']
    
    logger.debug(f"✅ 模拟交易完成，总交易日: {len(account_df)}")
    logger.debug(f"💰 总手续费: ￥{account_df['手续费'].sum():,.2f}")
    
    return account_df


def simulate_performance(select_results_path: Path,
                               data_path: Path,
                               runtime_folder:Path,
                               start_date: str,
                               end_date: str,
                               initial_cash: float = 1000000.0,
                               commission_rate: float = 0.0003,
                               stamp_tax_rate: float = 0.001,
                               rebalance_time: str = "open",
                               show_plot: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    策略表现模拟
    
    :param select_results_path: 选股结果文件路径
    :param data_path: 数据路径
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param initial_cash: 初始资金
    :param commission_rate: 佣金费率
    :param stamp_tax_rate: 印花税率
    :param rebalance_time: 调仓时间
    :param show_plot: 是否显示图表
    :return: (账户资金曲线, 策略评价, 年度收益, 月度收益, 季度收益)
    """
    logger.info("🚀 开始策略模拟...")
    
    # 加载数据
    select_results = load_select_results(select_results_path)
    stock_data = load_stock_data(runtime_folder)
    trading_dates = get_trading_dates(start_date, end_date, data_path)
    
    # 模拟交易
    account_df = simulate_trading(
        select_results, stock_data, trading_dates,
        initial_cash, commission_rate, stamp_tax_rate,
        rebalance_time
    )
    
    # 策略评价
    rtn, year_return, month_return, quarter_return = strategy_evaluate(
        account_df, net_col="净值", pct_col="涨跌幅"
    )
    
    # 保存结果
    result_folder = select_results_path.parent
    result_folder.mkdir(parents=True, exist_ok=True)
    
    save_performance(
        result_folder,
        资金曲线=account_df,
        策略评价=rtn,
        年度账户收益=year_return,
        季度账户收益=quarter_return,
        月度账户收益=month_return,
    )
    
    # 显示图表
    if show_plot:
        show_performance_plot(
            result_folder, select_results, account_df, rtn, year_return,data_path,start_date,end_date
        )
    
    logger.info("✅ 策略模拟完成!")
    
    return account_df, rtn, year_return, month_return, quarter_return

# 计算策略评价指标
def strategy_evaluate(equity, net_col="净值", pct_col="涨跌幅"):
    """
    回测评价函数
    :param equity: 资金曲线数据
    :param net_col: 资金曲线列名
    :param pct_col: 周期涨跌幅列名
    :return:
    """
    # ===新建一个dataframe保存回测指标
    results = pd.DataFrame()

    # 将数字转为百分数
    def num_to_pct(value):
        return "%.2f%%" % (value * 100)

    # ===计算累积净值
    results.loc[0, "累积净值"] = round(equity[net_col].iloc[-1], 2)

    # ===计算年化收益
    days = (equity["交易日期"].iloc[-1] - equity["交易日期"].iloc[0]) / timedelta(days=1)
    annual_return = (equity[net_col].iloc[-1]) ** (365 / days) - 1
    results.loc[0, "年化收益"] = num_to_pct(annual_return)

    # ===计算最大回撤，最大回撤的含义：《如何通过3行代码计算最大回撤》https://mp.weixin.qq.com/s/Dwt4lkKR_PEnWRprLlvPVw
    # 计算当日之前的资金曲线的最高点
    equity[f'{net_col.split("资金曲线")[0]}max2here'] = equity[net_col].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    equity[f'{net_col.split("资金曲线")[0]}dd2here'] = (
        equity[net_col] / equity[f'{net_col.split("资金曲线")[0]}max2here'] - 1
    )
    # 计算最大回撤，以及最大回撤结束时间
    end_date, max_draw_down = tuple(
        equity.sort_values(by=[f'{net_col.split("资金曲线")[0]}dd2here']).iloc[0][
            ["交易日期", f'{net_col.split("资金曲线")[0]}dd2here']
        ]
    )
    # 计算最大回撤开始时间
    start_date = equity[equity["交易日期"] <= end_date].sort_values(by=net_col, ascending=False).iloc[0]["交易日期"]
    results.loc[0, "最大回撤"] = num_to_pct(max_draw_down)
    results.loc[0, "最大回撤开始时间"] = str(start_date)
    results.loc[0, "最大回撤结束时间"] = str(end_date)
    # ===年化收益/回撤比：我个人比较关注的一个指标
    results.loc[0, "年化收益/回撤比"] = round(annual_return / abs(max_draw_down), 2)
    mean_back_zf = 1 / (1 + equity[f'{net_col.split("资金曲线")[0]}dd2here']) - 1  # 回本涨幅
    mean_fix_zf = mean_back_zf.mean()  # 修复涨幅
    max_back_zf = 1 / (1 + max_draw_down) - 1  # 回本涨幅
    max_fix_zf = max_back_zf.mean()  # 修复涨幅
    results.loc[0, "修复涨幅（均/最大）"] = f"{num_to_pct(mean_fix_zf)} / {num_to_pct(max_fix_zf)}"
    results.loc[0, "修复时间（均/最大）"] = (
        f"{round(np.log10(1 + mean_fix_zf) / np.log10(1 + annual_return) * 365, 1)} / "
        f"{round(np.log10(1 + max_fix_zf) / np.log10(1 + annual_return) * 365, 1)}"
    )
    # ===统计每个周期
    results.loc[0, "盈利周期数"] = len(equity.loc[equity[pct_col] > 0])  # 盈利笔数
    results.loc[0, "亏损周期数"] = len(equity.loc[equity[pct_col] <= 0])  # 亏损笔数
    not_zero = len(equity.loc[equity[pct_col] != 0])
    results.loc[0, "胜率（含0/去0）"] = (
        f"{num_to_pct(results.loc[0, '盈利周期数'] / len(equity))} / "
        f"{num_to_pct(len(equity.loc[equity[pct_col] > 0]) / not_zero)}"
    )  # 胜率
    results.loc[0, "每周期平均收益"] = num_to_pct(equity[pct_col].mean())  # 每笔交易平均盈亏
    results.loc[0, "盈亏收益比"] = round(
        equity.loc[equity[pct_col] > 0][pct_col].mean() / equity.loc[equity[pct_col] <= 0][pct_col].mean() * (-1), 2
    )  # 盈亏比

    results.loc[0, "单周期最大盈利"] = num_to_pct(equity[pct_col].max())  # 单笔最大盈利
    results.loc[0, "单周期大亏损"] = num_to_pct(equity[pct_col].min())  # 单笔最大亏损

    # ===连续盈利亏损
    results.loc[0, "最大连续盈利周期数"] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(equity[pct_col] > 0, 1, np.nan))]
    )  # 最大连续盈利次数
    results.loc[0, "最大连续亏损周期数"] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(equity[pct_col] <= 0, 1, np.nan))]
    )  # 最大连续亏损次数

    # ===其他评价指标
    results.loc[0, "收益率标准差"] = num_to_pct(equity[pct_col].std())

    # 空仓时，防止显示nan
    fillna_col = ["年化收益/回撤比", "盈亏收益比"]
    results[fillna_col] = results[fillna_col].fillna(0)

    # ===每年、每月收益率
    temp = equity.copy()
    temp.set_index("交易日期", inplace=True)

    year_return = temp[[pct_col]].resample(rule="YE").apply(lambda x: (1 + x).prod() - 1)
    month_return = temp[[pct_col]].resample(rule="ME").apply(lambda x: (1 + x).prod() - 1)
    quarter_return = temp[[pct_col]].resample(rule="QE").apply(lambda x: (1 + x).prod() - 1)

    def num2pct(x):
        if str(x) != "nan":
            return str(round(x * 100, 2)) + "%"
        else:
            return x

    year_return["涨跌幅"] = year_return[pct_col].apply(num2pct)
    month_return["涨跌幅"] = month_return[pct_col].apply(num2pct)
    quarter_return["涨跌幅"] = quarter_return[pct_col].apply(num2pct)

    return results.T, year_return, month_return, quarter_return
