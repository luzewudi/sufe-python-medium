import time
import warnings
from pathlib import Path

import pandas as pd

from core.select_stock import select_stocks, RES_COLS
from core.utils.log_kit import logger, divider
from core.utils.path_kit import get_folder_path
import config

# ====================================================================================================
# ** 配置与初始化 **
# 忽略警告并设定显示选项，以优化代码输出的可读性
# ====================================================================================================
warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)




def concat_select_results_simple(strategy_list: list, result_folder: str) -> pd.DataFrame:
    """
    聚合策略择时结果，形成综合择时结果
    根据每个策略的cap_weight重新分配资金占比，并合并相同股票的资金占比
    """
    all_select_df_list = []  # 存储每一个策略的择时结果
    recent_select_df_list = []
    
    # 计算总权重
    total_weight = sum(strategy.get("cap_weight", 1.0) for strategy in strategy_list)
    logger.debug(f"📊 策略总权重: {total_weight}")

    for strategy in strategy_list:
        strategy_name = strategy.get("name", "默认策略")
        strategy_weight = strategy.get("cap_weight", 1.0)
        stg_select_result = Path(result_folder) / f"选股结果_{strategy_name}.pkl"
        
        # 如果文件不存在，就跳过
        if not stg_select_result.exists():
            logger.warning(f"⚠️ 文件不存在: {stg_select_result}")
            continue
            
        try:
            # 读入单策略择时结果
            stg_select = pd.read_pickle(stg_select_result)
            if stg_select.empty:
                logger.warning(f"⚠️ 文件为空: {stg_select_result}")
                continue
                
            logger.debug(f"📊 读取 {strategy_name}: {stg_select.shape[0]} 行")
            
            # 基本数据清理
            stg_select = stg_select.dropna(subset=['选股日期', '股票代码', '股票名称']).copy()
            
            if stg_select.empty:
                logger.warning(f"⚠️ 清理后文件为空: {stg_select_result}")
                continue
            
            # 根据策略权重调整目标资金占比
            stg_select["目标资金占比"] = stg_select["目标资金占比"] * (strategy_weight / total_weight)
            
            # 添加策略标识
            stg_select["策略"] = strategy_name
            
            all_dataframes.append(stg_select)
            logger.debug(f"✅ 添加策略 {strategy_name}: {stg_select.shape[0]} 行")
            
        except Exception as e:
            logger.error(f"❌ 读取 {stg_select_result} 失败: {e}")
            continue

    if not all_dataframes:
        logger.warning("⚠️ 没有有效的数据文件")
        return pd.DataFrame(columns=RES_COLS)
    
    # 合并所有数据
    logger.info("🔄 合并所有策略数据...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"📊 合并后总行数: {combined_df.shape[0]}")
    combined_df["选股日期"] = pd.to_datetime(combined_df["选股日期"])
    combined_df["股票代码"] = combined_df["股票代码"].astype(str).str.strip()  # 同时去除空格
    # 按选股日期、股票代码分组，合并相同股票的资金占比和择时信号
    logger.info("🔄 合并相同股票的资金占比和择时信号...")
    before_merge = combined_df.shape[0]
    
    merged_df = combined_df.groupby(["选股日期", "股票代码"]).agg({
        "股票名称": "first",  # 取第一个股票名称
        "策略": lambda x: "综合策略",  # 合并后标记为综合策略
        "换仓时间": "first",  # 取第一个换仓时间
        "目标资金占比": "sum",  # 求和合并资金占比
        "择时信号": "max"  # 取最大值（任一策略有信号则为1）
    }).reset_index()
    
    after_merge = merged_df.shape[0]
    logger.info(f"🔄 合并完成: {before_merge} -> {after_merge} 行")
    
    # 重新排序列
    all_select_df = merged_df[RES_COLS].sort_values(by=["选股日期", "股票代码"]).reset_index(drop=True)
    
    # 生成最新数据
    if not all_select_df.empty:
        latest_date = all_select_df['选股日期'].max()
        recent_select_df = all_select_df[all_select_df['选股日期'] == latest_date].copy()
        logger.info(f"📅 最新数据: {recent_select_df.shape[0]} 行 (日期: {latest_date})")
    else:
        recent_select_df = pd.DataFrame(columns=RES_COLS)
    
    # 保存择时结果
    select_results_path = Path(result_folder) / "选股结果.pkl"
    all_select_df.to_pickle(select_results_path)
    all_select_df.to_csv(select_results_path.with_suffix(".csv"), encoding="utf-8-sig", index=False)
    recent_select_df.to_csv(Path(result_folder) / "最新选股结果.csv", encoding="utf-8-sig", index=False)

    logger.info(f"✅ 策略合并完成，最终股票数量: {len(all_select_df)}")
    return all_select_df


if __name__ == '__main__':

    # 获取路径
    runtime_folder = get_folder_path(config.runtime_data_path, "运行缓存", config.backtest_name)
    result_folder = get_folder_path(config.runtime_data_path, "回测结果", config.backtest_name)
    
    # 根据计算得到的因子进行择时
    divider('择时信号', '-')
    s_time = time.time()
    select_stocks(config.strategy_list, str(runtime_folder), str(result_folder), boost=False)
    select_results = concat_select_results_simple(config.strategy_list, str(result_folder))  # 合并多个策略的择时结果

    logger.debug(f'💾 择时结果数据大小：{select_results.memory_usage(deep=True).sum() / 1024 / 1024:.4f} MB')
    logger.ok(f'择时完成，总耗时：{time.time() - s_time:.3f}秒')
