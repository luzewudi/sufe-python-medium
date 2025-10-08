"""
因子计算模块
整合了因子配置、因子计算和因子存储功能
"""
import gc
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Union
from dataclasses import dataclass
from functools import cached_property
import re
import importlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.utils.log_kit import logger
from pathlib import Path
import config

def get_col_name(factor_name, factor_param):
    """生成因子列名"""
    col_name = f"{factor_name}"
    if factor_param:  # 如果参数有意义的话才显示出来
        if isinstance(factor_param, (tuple, list)):
            factor_param_str = "(" + ",".join(map(str, factor_param)) + ")"
        else:
            factor_param_str = str(factor_param)
        col_name += f"_{factor_param_str}"
    return col_name


# ====================================================================================================
# ** 因子配置相关类 **
# ====================================================================================================


# ====================================================================================================
# ** 因子接口和因子中心 **
# ====================================================================================================
# 因子接口类型别名，用于类型提示
FactorInterface = type('FactorInterface', (), {})


class FactorHub:
    _factor_cache = {}

    @staticmethod
    def get_by_name(factor_name):
        if factor_name in FactorHub._factor_cache:
            return FactorHub._factor_cache[factor_name]

        try:
            # 构造模块名
            module_name = f"因子库.{factor_name}"

            # 动态导入模块
            factor_module = importlib.import_module(module_name)

            # 创建一个包含模块变量和函数的字典
            factor_content = {
                name: getattr(factor_module, name) for name in dir(factor_module)
                if not name.startswith("__")
            }

            if 'fin_cols' not in factor_content:
                factor_content['fin_cols'] = []

            # 创建一个包含这些变量和函数的对象
            factor_instance = type(factor_name, (), factor_content)

            # 缓存策略对象
            FactorHub._factor_cache[factor_name] = factor_instance

            return factor_instance
        except ModuleNotFoundError:
            raise ValueError(f"Factor {factor_name} not found.")
        except AttributeError:
            raise ValueError(f"Error accessing factor content in module {factor_name}.")


# ====================================================================================================
# ** 因子计算相关常量 **
# ====================================================================================================
# 因子计算之后，需要保存的行情数据
FACTOR_COLS = [
    "交易日期",
    "股票代码",
    "股票名称",
    "上市至今交易天数",
    "复权因子",
    "开盘价",
    "最高价",
    "最低价",
    "收盘价",
    "成交额",
    "是否交易",
    "流通市值",
    "总市值",
    "下日_开盘涨停",
    "下日_是否ST",
    "下日_是否交易",
    "下日_是否退市",
    "新版申万一级行业名称"
]


# ====================================================================================================
# ** 因子计算核心函数 **
# ====================================================================================================
def cal_strategy_factors(
    factor_params_dict: dict,
    stock_code: str,
    candle_df: pd.DataFrame,
    fin_data: Dict[str, pd.DataFrame] = None,
    factor_col_name_list: List[str] = (),
    start_date: str = None,
    end_date: str = None,
):
    """
    计算指定股票的策略因子。

    参数:
    factor_params_dict (dict): 因子参数字典
    stock_code (str): 股票代码
    candle_df (DataFrame): 股票的K线数据，已经按照"交易日期"从小到大排序
    fin_data (dict): 财务数据
    factor_col_name_list (list): 需要计算的因子列名称列表
    start_date (str): 开始日期
    end_date (str): 结束日期

    返回:
    DataFrame: 包含计算因子的K线数据
    """
    factor_series_dict = {}
    before_len = len(candle_df)

    candle_df.sort_values(by="交易日期", inplace=True)  # 防止因子计算出错，计算之前，先进行排序
    for factor_name, param_list in factor_params_dict.items():
        factor_file = FactorHub.get_by_name(factor_name)
        for param in param_list:
            col_name = get_col_name(factor_name, param)
            if col_name in factor_col_name_list:
                # 因子计算，factor_df是包含因子计算结果的DataFrame，必须是按照"交易日期"从小到大排序
                factor_df = factor_file.add_factor(
                    candle_df.copy(),
                    param,
                    fin_data=fin_data,
                    col_name=col_name,
                )

                factor_series_dict[col_name] = factor_df[col_name].values
                # 检查因子计算是否出错
                if before_len != len(factor_series_dict[col_name]):
                    logger.error(
                        f"{stock_code}的{factor_name}因子({param}，{col_name})导致数据长度发生变化，请检查！"
                    )
                    raise Exception("因子计算出错，请避免在cal_factors中修改数据行数")

    kline_with_factor_dict = {**{col_name: candle_df[col_name] for col_name in FACTOR_COLS}, **factor_series_dict}
    kline_with_factor_df = pd.DataFrame(kline_with_factor_dict)
    kline_with_factor_df.sort_values(by="交易日期", inplace=True)

    # 根据回测设置的时间区间进行裁切
    start_date = start_date or kline_with_factor_df["交易日期"].min()
    end_date = end_date or kline_with_factor_df["交易日期"].max()
    date_cut_condition = (kline_with_factor_df["交易日期"] >= start_date) & (
        kline_with_factor_df["交易日期"] <= end_date
    )

    return kline_with_factor_df[date_cut_condition].reset_index(drop=True)  # 返回计算完的因子数据


def process_by_stock(
    factor_params_dict: dict,
    candle_df: pd.DataFrame,
    factor_col_name_list: List[str],
    idx: int,
    fin_cols: List[str] = None,
    start_date: str = None,
    end_date: str = None,
):
    """
    组装因子计算必要的数据结构，并且送入到因子计算函数中进行计算
    :param factor_params_dict: 因子参数字典
    :param candle_df: 单只股票的K线数据
    :param factor_col_name_list: 需要计算的因子列名称列表
    :param idx: 股票索引
    :param fin_cols: 财务列列表
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: idx, factor_df
    """
    stock_code = candle_df.iloc[-1]["股票代码"]
    # 导入财务数据，将个股数据与财务数据合并，并计算财务指标的衍生指标
    if fin_cols:  # 前面已经做了预检，这边只需要动态加载即可
        # 分别为：个股数据、财务数据、原始财务数据（不抛弃废弃的报告数据）
        # 简化版本，暂时跳过财务数据合并
        fin_data = None
    else:
        fin_data = None

    factor_df = cal_strategy_factors(
        factor_params_dict, stock_code, candle_df, fin_data, factor_col_name_list, start_date, end_date
    )

    return idx, factor_df


def calculate_factors(
    runtime_folder: str,
    factor_params_dict: dict,
    factor_col_name_list: List[str],
    fin_cols: List[str] = None,
    start_date: str = None,
    end_date: str = None,
):
    """
    计算所有股票的因子（默认使用多线程）
    
    参数:
    runtime_folder (str): 运行时文件夹路径
    factor_params_dict (dict): 因子参数字典
    factor_col_name_list (list): 需要计算的因子列名称列表
    fin_cols (list): 财务列列表
    start_date (str): 开始日期
    end_date (str): 结束日期
    """
    logger.info("因子计算...")
    s_time = time.time()

    # 1. 加载股票K线数据
    logger.debug("💿 读取股票K线数据...")
    candle_df_dict: Dict[str, pd.DataFrame] = pd.read_pickle(Path(runtime_folder) / "股票预处理数据.pkl")

    # 2. 计算因子并存储结果
    factor_col_count = len(factor_col_name_list)
    shards = range(0, factor_col_count, config.factor_col_limit)

    logger.debug(f"* 总共计算因子个数：{factor_col_count} 个")
    logger.debug(f"* 单次计算因子个数：{config.factor_col_limit} 个，(需分成{len(shards)}组计算)")
    logger.debug(f"* 需要计算股票数量：{len(candle_df_dict.keys())} 个")
    logger.debug(f"🚀 多进程计算因子，进程数量：{config.n_jobs}")

    # 清理 cache 的缓存
    all_kline_pkl = Path(runtime_folder) / "all_factors_kline.pkl"
    all_kline_pkl.unlink(missing_ok=True)

    logger.debug(f"🚀 多进程计算因子，进程数量：{config.n_jobs}" )
    for shard_index in shards:
        logger.debug(f"🗂️ 因子分片计算中，进度：{int(shard_index / config.factor_col_limit) + 1}/{len(shards)}")
        factor_col_name_list_shard = factor_col_name_list[shard_index : shard_index + config.factor_col_limit]
        all_factor_df_list = [pd.DataFrame()] * len(candle_df_dict.keys())

        # 使用多进程计算因子
        with ProcessPoolExecutor(max_workers=config.n_jobs) as executor:
            futures = []
            for candle_idx, candle_df in enumerate(candle_df_dict.values()):
                futures.append(
                    executor.submit(
                        process_by_stock,
                        factor_params_dict,
                        candle_df,
                        factor_col_name_list_shard,
                        candle_idx,
                        fin_cols,
                        start_date,
                        end_date,
                    )
                )

            for future in tqdm(futures, desc="🧮 计算因子", total=len(futures), mininterval=2, file=sys.stdout):
                try:
                    idx, period_df = future.result()
                    all_factor_df_list[idx] = period_df
                except Exception as e:
                    logger.error(f"因子计算失败：{e}")
                    logger.debug(traceback.format_exc())
                    raise e

        # 3. 合并因子数据并存储
        all_factors_df = pd.concat(all_factor_df_list, ignore_index=True, copy=False)
        logger.debug("📅 因子结果最晚日期：" + str(all_factors_df["交易日期"].max()))

        # 转化一下symbol的类型为category，可以加快因子计算速度，节省内存
        # 并且排序和整理index
        all_factors_df = (
            all_factors_df.assign(
                股票代码=all_factors_df["股票代码"].astype("category"),
                股票名称=all_factors_df["股票名称"].astype("category"),
            )
            .sort_values(by=["交易日期", "股票代码"])
            .reset_index(drop=True)
        )

        logger.debug("💾 存储因子数据...")

        # 存储选股需要的k线数据
        if not all_kline_pkl.exists():
            all_kline_df = all_factors_df[FACTOR_COLS].sort_values(by=["交易日期", "股票代码", "股票名称"])
            all_kline_df.to_pickle(all_kline_pkl)

        # 存储每个因子的数据
        for factor_col_name in factor_col_name_list_shard:
            factor_pkl = Path(runtime_folder) / f"factor_{factor_col_name}.pkl"
            factor_pkl.unlink(missing_ok=True)
            all_factors_df[factor_col_name].to_pickle(factor_pkl)

        #存储大作业需要看的dataframe
        if shard_index == 0:  # 只在第一个分片时创建任务一df
            # 选择需要的列
            required_cols = ["交易日期", "股票代码", "股票名称", "开盘价", "最高价", "最低价", "收盘价", "总市值"]
            task_df_cols = required_cols + factor_col_name_list_shard
            
            # 创建任务一df
            task_df = all_factors_df[task_df_cols].copy()
            
            # 计算每个因子的统计信息
            factor_stats = {}
            for factor_col in factor_col_name_list_shard:
                if factor_col in task_df.columns:
                    factor_data = task_df[factor_col].dropna()
                    if len(factor_data) > 0:
                        factor_stats[factor_col] = {
                            '计数': len(factor_data),
                            '平均值': factor_data.mean(),
                            '标准差': factor_data.std(),
                            '最小值': factor_data.min(),
                            '20%分位数': factor_data.quantile(0.2),
                            '50%分位数': factor_data.quantile(0.5),
                            '75%分位数': factor_data.quantile(0.75),
                            '最大值': factor_data.max()
                        }
            
            # 保存任务一df
            task_df_path = Path(runtime_folder) / "任务一df.pkl"
            task_df.to_pickle(task_df_path)
            logger.debug(f"💾 保存任务一df到：{task_df_path}")
            
            # 保存因子统计信息
            factor_stats_df = pd.DataFrame(factor_stats).T
            factor_stats_path = Path(runtime_folder) / "因子统计信息.csv"
            factor_stats_df.to_csv(factor_stats_path, encoding='utf-8-sig')
            logger.debug(f"💾 保存因子统计信息到：{factor_stats_path}")
            
            # 创建因子统计信息可视化
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
                plt.rcParams['axes.unicode_minus'] = False  # 支持负号显示
                
                if factor_stats:
                    # 设置图形大小
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    fig.suptitle('因子统计信息可视化', fontsize=16, fontweight='bold')
                    
                    # 1. 因子平均值对比柱状图
                    factor_names = list(factor_stats.keys())[:min(8, len(factor_stats))]
                    mean_values = [factor_stats[name]['平均值'] for name in factor_names]
                    axes[0, 0].bar(range(len(factor_names)), mean_values, alpha=0.7, edgecolor='black')
                    axes[0, 0].set_title('因子平均值对比')
                    axes[0, 0].set_xlabel('因子名称')
                    axes[0, 0].set_ylabel('平均值')
                    axes[0, 0].set_xticks(range(len(factor_names)))
                    axes[0, 0].set_xticklabels(factor_names, rotation=45)
                    
                    # 2. 因子标准差对比柱状图
                    std_values = [factor_stats[name]['标准差'] for name in factor_names]
                    axes[0, 1].bar(range(len(factor_names)), std_values, alpha=0.7, edgecolor='black', color='orange')
                    axes[0, 1].set_title('因子标准差对比')
                    axes[0, 1].set_xlabel('因子名称')
                    axes[0, 1].set_ylabel('标准差')
                    axes[0, 1].set_xticks(range(len(factor_names)))
                    axes[0, 1].set_xticklabels(factor_names, rotation=45)
                    
                    # 3. 因子相关性热力图（选择前几个因子）
                    if len(factor_col_name_list_shard) > 1:
                        factor_corr_data = task_df[factor_col_name_list_shard[:min(10, len(factor_col_name_list_shard))]].corr()
                        sns.heatmap(factor_corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
                        axes[1, 0].set_title('因子相关性热力图')
                    else:
                        axes[1, 0].text(0.5, 0.5, '因子数量不足，无法生成相关性热力图', 
                                       ha='center', va='center', transform=axes[1, 0].transAxes)
                        axes[1, 0].set_title('因子相关性热力图')
                    
                    # 4. 因子分布箱线图（选择前几个因子）
                    if len(factor_col_name_list_shard) > 0:
                        factor_data_for_box = []
                        factor_labels = []
                        for factor_col in factor_col_name_list_shard[:min(6, len(factor_col_name_list_shard))]:
                            if factor_col in task_df.columns:
                                factor_data = task_df[factor_col].dropna()
                                if len(factor_data) > 0:
                                    factor_data_for_box.append(factor_data)
                                    factor_labels.append(factor_col)
                        
                        if factor_data_for_box:
                            axes[1, 1].boxplot(factor_data_for_box, labels=factor_labels)
                            axes[1, 1].set_title('因子分布箱线图')
                            axes[1, 1].set_xlabel('因子名称')
                            axes[1, 1].set_ylabel('因子值')
                            axes[1, 1].tick_params(axis='x', rotation=45)
                        else:
                            axes[1, 1].text(0.5, 0.5, '无有效因子数据', ha='center', va='center', transform=axes[1, 1].transAxes)
                            axes[1, 1].set_title('因子分布箱线图')
                    else:
                        axes[1, 1].text(0.5, 0.5, '无因子数据', ha='center', va='center', transform=axes[1, 1].transAxes)
                        axes[1, 1].set_title('因子分布箱线图')
                    
                    plt.tight_layout()
                    
                    # 保存图片
                    visualization_path = Path(runtime_folder) / "任务一df可视化.png"
                    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.debug(f"💾 保存因子统计信息可视化图片到：{visualization_path}")
                else:
                    logger.warning("无因子统计信息，跳过可视化生成")
                
            except ImportError:
                logger.warning("matplotlib或seaborn未安装，跳过可视化生成")
            except Exception as e:
                logger.warning(f"可视化生成失败：{e}")

        gc.collect()

    logger.ok(f"因子计算完成，耗时：{time.time() - s_time:.2f}秒")
