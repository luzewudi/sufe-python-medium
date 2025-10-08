import gc
import sys
import time
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import n_jobs
from core.utils.log_kit import logger
from core.utils.path_kit import get_file_path

# 择时相关的常量
KLINE_COLS = ["交易日期", "股票代码", "股票名称"]
# 计算完择时之后，保留的字段
RES_COLS = [
    "选股日期",
    "股票代码",
    "股票名称",
    "策略",
    "换仓时间",
    "目标资金占比",
    "择时信号",
]


def get_strategy_by_name(strategy_name: str) -> dict:
    """
    根据策略名称动态加载策略库中的策略文件
    
    参数:
    strategy_name: 策略名称
    
    返回:
    dict: 策略模块中的函数字典
    """
    try:
        # 构造模块名
        module_name = f"策略库.{strategy_name}"
        
        # 动态导入模块
        strategy_module = importlib.import_module(module_name)
        
        # 创建一个包含模块变量和函数的字典
        strategy_content = {
            name: getattr(strategy_module, name)
            for name in dir(strategy_module)
            if not name.startswith("__") and callable(getattr(strategy_module, name))
        }
        
        return strategy_content
    except ModuleNotFoundError:
        return {}
    except AttributeError as e:
        logger.error(f"访问策略模块 {strategy_name} 时出错: {e}")
        return {}


def validate_strategy_exists(strategy_name: str) -> bool:
    """
    验证策略库中是否存在指定名称的策略文件
    
    参数:
    strategy_name: 策略名称
    
    返回:
    bool: 策略是否存在
    """
    strategy_content = get_strategy_by_name(strategy_name)
    return len(strategy_content) > 0


# ================================================================
# step3_择时.py
# ================================================================
def select_stocks(strategy_list: List[dict], runtime_folder: str, result_folder: str, boost=True):
    """
    择时主函数
    
    参数:
    strategy_list: 策略列表
    runtime_folder: 运行时文件夹路径
    result_folder: 结果文件夹路径
    boost: 是否使用多进程
    """
    if boost:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(select_stock_by_strategy, strategy, runtime_folder, result_folder, silent=False) for strategy in strategy_list]
            for future in tqdm(as_completed(futures), total=len(strategy_list), desc="择时", mininterval=2, file=sys.stdout):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(e)
                    sys.exit(1)
    else:
        for strategy in tqdm(strategy_list, total=len(strategy_list), desc="择时", mininterval=2, file=sys.stdout):
            select_stock_by_strategy(strategy, runtime_folder, result_folder, silent=False)

    import logging
    logger.setLevel(logging.DEBUG)  # 恢复日志模式


def select_stock_by_strategy(strategy: dict, runtime_folder: str, result_folder: str, silent=False):
    """
    择时流程：
    1. 初始化策略配置
    2. 加载并清洗数据
    3. 应用过滤函数过滤股票池
    4. 计算择时信号
    5. 缓存择时结果

    参数:
    strategy: 策略配置字典
    runtime_folder: 运行时文件夹路径
    result_folder: 结果文件夹路径
    silent: 是否静默模式
    返回:
    DataFrame: 择时结果
    """
    if silent:
        import logging
        logger.setLevel(logging.WARNING)  # 可以减少中间输出的log


    factor_df_path = Path(runtime_folder) / "all_factors_kline.pkl"
    logger.debug(f"🔍 因子文件：{factor_df_path}")
    
    timing_stocks_by_strategy(strategy, factor_df_path, result_folder)


def timing_stocks_by_strategy(strategy: dict, factor_df_path, result_folder: str):
    # ====================================================================================================
    # 1. 初始化策略配置
    # ====================================================================================================
    s_time = time.time()
    strategy_name = strategy.get("name")
    logger.debug(f"🎯 {strategy_name} 择时启动...")
    
    # 验证策略是否存在
    if not validate_strategy_exists(strategy_name):
        logger.error(f"❌ 策略库中未找到策略文件: {strategy_name}")
        logger.error(f"请检查策略库文件夹中是否存在 {strategy_name}.py 文件")
        return
    
    # 加载策略模块
    strategy_content = get_strategy_by_name(strategy_name)
    if not strategy_content:
        logger.error(f"❌ 无法加载策略模块: {strategy_name}")
        return

    # ====================================================================================================
    # 2. 加载并清洗数据
    # ====================================================================================================
    # 准备择时用数据
    runtime_folder = factor_df_path.parent
    factor_df = pd.read_pickle(factor_df_path)
    
    # 获取因子列名（包括过滤因子和择时因子）
    factor_columns = []
    
    # 处理所有因子（过滤因子和择时因子）
    all_factors = strategy.get("filter_list", []) + strategy.get("timing_list", [])
    for factor in all_factors:
        factor_name = factor[0]
        param = factor[2] if len(factor) > 2 else None
        
        # 生成列名
        col_name = f"{factor_name}"
        if param:
            if isinstance(param, (tuple, list)):
                param_str = "(" + ",".join(map(str, param)) + ")"
            else:
                param_str = str(param)
            col_name += f"_{param_str}"
        
        factor_columns.append(col_name)
    
    # 加载因子数据
    for factor_col_name in factor_columns:
        factor_df[factor_col_name] = pd.read_pickle(get_file_path(runtime_folder, f"factor_{factor_col_name}.pkl"))
    logger.debug(f'📦 [{strategy_name}] 择时数据加载完成，最晚日期：{factor_df["交易日期"].max()}')

    # 过滤掉每一个周期中，没有交易的股票
    factor_df = factor_df[(factor_df["是否交易"] == 1)].dropna(subset=factor_columns).copy()
    factor_df.dropna(subset=["股票代码"], inplace=True)

    # 最后整理一下
    factor_df.sort_values(by=["交易日期", "股票代码"], inplace=True)
    factor_df.reset_index(drop=True, inplace=True)

    logger.debug(f'➡️ [{strategy_name}] 数据清洗完成，去掉空因子数据，最晚日期：{factor_df["交易日期"].max()}')

    # ====================================================================================================
    # 3. 应用过滤函数过滤股票池
    # ====================================================================================================
    s = time.time()
    
    try:
        # 从策略模块中获取过滤函数
        filter_stock_func = strategy_content.get('filter_stock', None)

        if filter_stock_func:
            # 直接传递策略字典
            factor_df = filter_stock_func(factor_df, strategy)
            logger.debug(f'🔍 [{strategy_name}] 过滤函数应用完成，剩余股票数：{len(factor_df)}')
        else:
            logger.warning(f'⚠️ [{strategy_name}] 策略文件中未找到过滤函数，跳过过滤步骤')
    except Exception as e:
        logger.warning(f'⚠️ [{strategy_name}] 过滤函数执行失败：{e}，跳过过滤步骤')

    logger.debug(f"➡️ [{strategy_name}] 股票池过滤耗时：{time.time() - s:.2f}s")

    # ====================================================================================================
    # 4. 计算择时信号
    # ====================================================================================================
    s = time.time()
    
    try:
        # 从策略模块中获取择时函数
        calc_timing_factor_func = strategy_content.get('calc_timing_factor', None)
        
        if calc_timing_factor_func:
            # 直接传递策略字典
            factor_df = calc_timing_factor_func(factor_df, strategy)
            logger.debug(f'⏰ [{strategy_name}] 择时信号计算完成')
        else:
            logger.warning(f'⚠️ [{strategy_name}] 策略文件中未找到择时函数，使用默认择时信号')
            # 默认择时信号为0（空仓）
            factor_df['择时信号'] = 0.0
    except Exception as e:
        logger.warning(f'⚠️ [{strategy_name}] 择时函数执行失败：{e}，使用默认择时信号')
        factor_df['择时信号'] = 0.0

    logger.debug(f"➡️ [{strategy_name}] 择时信号计算耗时：{time.time() - s:.2f}s")

    # ====================================================================================================
    # 5. 生成择时结果
    # ====================================================================================================
    s = time.time()
    
    # 导入严格模式配置
    from config import strict_mode
    
    # 根据严格模式进行资金分配
    if strict_mode:
        # 严格模式：所有过滤后的股票均分资金，但择时信号为0的股票要乘以0
        # 这样目标资金占比加起来不为1（有闲置资金）
        factor_df['目标资金占比'] = 1.0 / factor_df.groupby('交易日期')['股票代码'].transform('size')
        # 择时信号为0的股票乘以0
        factor_df.loc[factor_df['择时信号'] == 0.0, '目标资金占比'] *= 0.0
        logger.debug(f'🔧 [{strategy_name}] 使用严格模式：所有股票均分资金，择时信号为0的股票乘以0（有闲置资金）')
    else:
        # 非严格模式：只在择时信号为1的股票中分配资金
        # 目标资金占比加起来为1（无闲置资金）
        timing_1_stocks = factor_df[factor_df['择时信号'] == 1.0].copy()
        if not timing_1_stocks.empty:
            # 计算择时信号为1的股票数量
            timing_1_count = timing_1_stocks.groupby('交易日期')['股票代码'].transform('size')
            # 只给择时信号为1的股票分配资金
            factor_df['目标资金占比'] = 0.0  # 默认为0
            factor_df.loc[factor_df['择时信号'] == 1.0, '目标资金占比'] = 1.0 / timing_1_count
        else:
            # 如果没有择时信号为1的股票，所有股票资金占比为0
            factor_df['目标资金占比'] = 0.0
        logger.debug(f'🔧 [{strategy_name}] 使用非严格模式：只在择时信号为1的股票中分配资金（无闲置资金）')
    
    # 准备结果数据
    result_df = factor_df[KLINE_COLS + ['目标资金占比', '择时信号']].copy()
    
    # 过滤掉空值行，确保数据质量
    result_df = result_df.dropna(subset=["股票代码", "股票名称"]).copy()
    # 过滤掉空字符串
    result_df = result_df[
        (result_df["股票代码"].str.strip() != "") & 
        (result_df["股票名称"].str.strip() != "")
    ].copy()
    
    result_path = Path(result_folder) / f"选股结果_{strategy_name}.pkl"
    
    # 若无择时结果则直接返回
    if result_df.empty:
        pd.DataFrame(columns=RES_COLS).to_pickle(result_path)
        return

    # ====================================================================================================
    # 6. 缓存择时结果
    # ====================================================================================================
    result_df = result_df.assign(
        策略=strategy_name, 
        策略权重=np.float64(strategy.get("cap_weight", 1.0)), 
        换仓时间=strategy.get("rebalance_time", "open")
    ).rename(columns={"交易日期": "选股日期"})

    result_df = result_df.assign(
        策略=result_df["策略"].astype("category"),
        换仓时间=result_df["换仓时间"].astype("category"),
        目标资金占比_原始=result_df["目标资金占比"],
        目标资金占比=(
            result_df["目标资金占比"]
            * result_df["择时信号"]
            * result_df["策略权重"]
        ).astype(np.float64),
    )

    # 缓存到本地文件
    result_df = result_df[RES_COLS]
    result_df.to_pickle(result_path)

    logger.debug(f"🏁 [{strategy_name}] 择时耗时: {(time.time() - s_time):.2f}s")

    return result_df


