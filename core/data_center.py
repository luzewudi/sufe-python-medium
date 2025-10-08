import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Union, Callable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from config import n_jobs
from core.market_essentials import cal_fuquan_price, cal_zdt_price, merge_with_index_data, import_index_data
from core.utils.log_kit import logger

# 定义股票数据所需的列
DATA_COLS = [
    "股票代码",
    "股票名称",
    "交易日期",
    "开盘价",
    "最高价",
    "最低价",
    "收盘价",
    "前收盘价",
    "成交量",
    "成交额",
    "流通市值",
    "总市值",
    "沪深300成分股",
    '新版申万一级行业名称'
]


# ================================================================
# step1_整理数据.py
# ================================================================
def prepare_data(
    stock_data_path: Path,
    index_data_path: Path,
    runtime_folder: Path,
    rebalance_time_list: List[str],
    boost: bool = True
):
    """
    准备股票数据，不依赖BacktestConfig类
    
    参数:
    stock_data_path: 股票数据路径
    index_data_path: 指数数据路径
    runtime_folder: 运行时缓存文件夹
    rebalance_time_list: 调仓时间列表
    boost: 是否使用多进程加速
    """
    logger.info(f"读取数据中心数据...")
    start_time = time.time()  # 记录数据准备开始时间
    
    # 确保目录存在
    runtime_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. 获取股票代码列表
    stock_code_list = []  # 用于存储股票代码
    # 遍历文件夹下，所有csv文件
    for filename in stock_data_path.glob("*.csv"):
        # 排除隐藏文件
        if filename.stem.startswith("."):
            continue
        stock_code_list.append(filename.stem)
    stock_code_list = sorted(stock_code_list)
    logger.debug(f"📂 读取到股票数量：{len(stock_code_list)}")

    # 2. 读取并处理指数数据，确保股票数据与指数数据的时间对齐
    index_data = import_index_data(index_data_path / "sh000001.csv", ["2007-01-01", None])
    all_candle_data_dict = {}  # 用于存储所有股票的K线数据

    logger.debug(f"🚀 多进程处理数据，进程数量：{n_jobs}" if boost else "🚲 单进程处理数据")
    if boost:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for code in stock_code_list:
                file_path = stock_data_path / f"{code}.csv"
                futures.append(executor.submit(prepare_data_by_stock, file_path, index_data, rebalance_time_list))

            for future in tqdm(futures, desc="📦 处理数据", total=len(futures), mininterval=2, file=sys.stdout):
                df = future.result()
                if not df.empty:
                    code = df["股票代码"].iloc[0]
                    all_candle_data_dict[code] = df  # 仅存储非空数据
    else:
        for code in tqdm(
            stock_code_list, desc="📦 处理数据", total=len(stock_code_list), mininterval=2, file=sys.stdout
        ):
            file_path = stock_data_path / f"{code}.csv"
            df = prepare_data_by_stock(file_path, index_data, rebalance_time_list)
            if not df.empty:
                all_candle_data_dict[code] = df

    # 获取所有股票数据的最大日期
    max_candle_date = max([df["交易日期"].max() for df in all_candle_data_dict.values()])

    # 3. 缓存预处理后的数据
    cache_path = runtime_folder / "股票预处理数据.pkl"
    logger.debug(f"📈 保存股票预处理数据: {cache_path}")
    logger.debug(f"📅 行情数据最新交易日期：{max_candle_date}")
    pd.to_pickle(all_candle_data_dict, cache_path)

    # 4. 准备并缓存pivot透视表数据，用于后续回测
    logger.debug("📄 生成行情数据透视表...")
    market_pivot_dict = make_market_pivot(all_candle_data_dict, rebalance_time_list)
    pivot_cache_path = runtime_folder / "全部股票行情pivot.pkl"
    logger.debug(f"🗄️ 保存行情数据透视表: {pivot_cache_path}")
    pd.to_pickle(market_pivot_dict, pivot_cache_path)

    logger.ok(f"数据准备耗时：{(time.time() - start_time):.2f} 秒")


def prepare_data_by_stock(
    stock_file_path: Union[str, Path], index_data: pd.DataFrame, rebalance_time_list: List[str]
) -> pd.DataFrame:
    """
    对股票数据进行预处理，包括合并指数数据和计算未来交易日状态。

    参数:
    stock_file_path (str | Path): 股票日线数据的路径
    index_data (DataFrame): 指数数据
    rebalance_time_list (List[str]): 调仓时间列表

    返回:
    df (DataFrame): 预处理后的数据
    """
    # 计算涨跌幅、换手率等关键指标
    df = pd.read_csv(
        stock_file_path, encoding="gbk", skiprows=1, parse_dates=["交易日期"], usecols=DATA_COLS
    )
    pct_change = df["收盘价"] / df["前收盘价"] - 1
    turnover_rate = df["成交额"] / df["流通市值"]
    trading_days = df.index.astype("int") + 1
    avg_price = df["成交额"] / df["成交量"]

    # 一次性赋值提高性能
    df = df.assign(涨跌幅=pct_change, 换手率=turnover_rate, 上市至今交易天数=trading_days, 均价=avg_price)

    # 复权价计算及涨跌停价格计算
    df = cal_fuquan_price(df, fuquan_type="后复权")
    df = cal_zdt_price(df)

    # 合并股票与指数数据，补全停牌日期等信息
    df = merge_with_index_data(df, index_data.copy(), fill_0_list=["换手率"])

    # 股票退市时间小于指数开始时间，就会出现空值
    if df.empty:
        # 如果出现这种情况，返回空的DataFrame用于后续操作
        return pd.DataFrame(columns=[*DATA_COLS, *rebalance_time_list])

    # 计算开盘买入涨跌幅和未来交易日状态
    df = df.assign(
        下日_是否交易=df["是否交易"].astype("int8").shift(-1),
        下日_一字涨停=df["一字涨停"].astype("int8").shift(-1),
        下日_开盘涨停=df["开盘涨停"].astype("int8").shift(-1),
        下日_是否ST=df["股票名称"].str.contains("ST").astype("int8").shift(-1),
        下日_是否S=df["股票名称"].str.contains("S").astype("int8").shift(-1),
        下日_是否退市=df["股票名称"].str.contains("退").astype("int8").shift(-1),
    )

    # 处理最后一根K线的数据：最后一根K线默认沿用前一日的数据
    state_cols = ["下日_是否交易", "下日_是否ST", "下日_是否S", "下日_是否退市"]
    df[state_cols] = df[state_cols].ffill()

    return df



def make_market_pivot(market_dict, rebalance_time_list):
    """
    构建市场数据的pivot透视表，便于回测计算。

    参数:
    market_dict (dict): 股票K线数据字典
    rebalance_time_list (list):分钟数据的字段列表

    返回:
    dict: 包含开盘价、收盘价及前收盘价的透视表数据
    """
    # cols = ["交易日期", "股票代码", "开盘价", "收盘价", "前收盘价", *rebalance_time_list]
    # counts = 3 + len(rebalance_time_list)
    cols = ["交易日期", "股票代码", "开盘价", "收盘价", "前收盘价"]
    counts = 3
    count = 1

    logger.debug("⚗️ 合成整体市场数据...")
    df_list = [df[cols].dropna(subset="股票代码") for df in market_dict.values()]
    df_all_market = pd.concat(df_list, ignore_index=True)
    col_names = {"开盘价": "open", "收盘价": "close", "前收盘价": "preclose"}

    markets = {}
    for col in cols[2:]:
        logger.debug(f"[{count}/{counts}] {col}透视表...")
        df_col = df_all_market.pivot(values=col, index="交易日期", columns="股票代码")
        markets[col_names.get(col, col)] = df_col
        count += 1

    return markets

