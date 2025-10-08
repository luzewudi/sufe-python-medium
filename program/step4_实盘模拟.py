import warnings
import pandas as pd
from pathlib import Path

from core.equity import simulate_performance
from core.utils.log_kit import divider, logger
from config import *

# ====================================================================================================
# ** 配置与初始化 **
# 忽略不必要的警告并设置显示选项，以优化控制台输出的可读性
# ====================================================================================================
warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

if __name__ == '__main__':
    logger.info("🚀 开始实盘模拟...")
    
    # 从config中读取配置参数
    select_results_path = get_folder_path(runtime_data_path, "回测结果", backtest_name,"选股结果.pkl")
    data_path = Path(data_center_path)
    runtime_folder=get_folder_path(runtime_data_path, "运行缓存",backtest_name)
    # 从策略配置中获取调仓时间（如果策略配置中有的话）
    if strategy_list:
        strategy_config = strategy_list[0]
        strategy_rebalance_time = strategy_config.get('rebalance_time')
        if strategy_rebalance_time:
            rebalance_time = strategy_rebalance_time
            logger.debug(f"📊 使用策略配置中的调仓时间: {strategy_config['name']} -> {rebalance_time}")
    
    logger.info(f"💰 初始资金: ￥{initial_cash:,.2f}")
    logger.info(f"📊 佣金费率: {c_rate*100:.3f}%")
    logger.info(f"📊 印花税率: {t_rate*100:.3f}%")
    logger.info(f"⏰ 调仓时间: {rebalance_time}")
    logger.info(f"📅 回测区间: {start_date} ~ {end_date}")
    logger.info(f"📁 选股结果文件: {select_results_path}")
    
    divider('模拟交易', '-')
    
    try:
        # 执行模拟
        account_df, rtn, year_return, month_return, quarter_return = simulate_performance(
            select_results_path=select_results_path,
            data_path=data_path,
            runtime_folder=runtime_folder,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            commission_rate=c_rate,
            stamp_tax_rate=t_rate,
            rebalance_time=rebalance_time,
            show_plot=True
        )
        
        # 输出最终结果
        final_net_value = account_df['净值'].iloc[-1]
        total_return = (final_net_value - 1) * 100
        max_drawdown = rtn.at["最大回撤", 0]
        annual_return = rtn.at["年化收益", 0]
        calmar_ratio = rtn.at["年化收益/回撤比", 0]
        
        logger.info("=" * 60)
        logger.info("📊 策略模拟结果汇总")
        logger.info("=" * 60)
        logger.info(f"💰 最终净值: {final_net_value:.4f}")
        logger.info(f"📈 总收益率: {total_return:.2f}%")
        logger.info(f"📊 年化收益率: {annual_return}")
        logger.info(f"📉 最大回撤: {max_drawdown}")
        logger.info(f"⚖️ 收益回撤比: {calmar_ratio}")
        logger.info(f"💸 总手续费: ￥{account_df['手续费'].sum():,.2f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 模拟交易失败: {e}")
        raise
