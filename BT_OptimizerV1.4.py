#2024-04-07     本地回测溢价率偏离+正股DIFF策略
#2024-05-11     1.  加入止盈
#               2.  本地回测溢价率偏离2.0 双刀头，+转债正股五日涨跌幅差
#2024-05-28     用OpTuna搜索最优参数
#2024-05-29     多进程搜索
#2024-05-30     V1.3 增加因子：正股年化波动率，成交额，到期收益率，转债市占比
#2024-5-30      V1.4 改为双目标:CAGR和SHARPE


#import warnings
#warnings.filterwarnings('ignore') # 忽略警告
import warnings
# 忽略Optuna分类分布警告
warnings.filterwarnings('ignore', category=UserWarning)
#warnings.filterwarnings('ignore', message="Choices for a categorical distribution should be a tuple of None, bool, int, float and str for persistent storage but contains")
#warnings.filterwarnings("ignore", category=UserWarning, module='optuna.distributions')

from datetime import datetime, timedelta
start_time = datetime.now()

#import matplotlib
#matplotlib.use('TkAgg')

import pandas as pd
from pandas import IndexSlice as idx
pd.set_option('display.max_columns', None)  # 当列太多时不换行

from numpy import exp, nan
#import akshare as ak
#import talib
#import time
#from LidoDBClass import LidoStockData

import itertools
import optuna
import copy
from joblib import Parallel, delayed
import quantstats as qs

factors = {
    'conv_bias15': 'conv_bias15',
    'pct_chg_gap5': 'pct_chg_gap5',
    'volatility': 'volatility',
    'dblow': 'dblow',
    'remain_cap': 'remain_cap',
    'bond_prem': 'bond_prem',
    'conv_prem': 'conv_prem',
    'mod_conv_prem': 'mod_conv_prem',
    'theory_bias': 'theory_bias',
    'volatility_stk': 'volatility_stk',
    'amount': 'amount',
    'ytm': 'ytm',
    'cap_mv_rate': 'cap_mv_rate',
    'turnover': 'turnover',
    'pct_chg_stk': 'pct_chg_stk'
}

factor_direction = {
    'conv_bias15': False,
    'pct_chg_gap5': False,
    'volatility': True,
    'dblow': False,
    'remain_cap': False,
    'bond_prem': False,
    'conv_prem': False,
    'mod_conv_prem': False,
    'theory_bias': False,
    'volatility_stk': True,
    'amount': True,
    'ytm': True,
    'cap_mv_rate': False,
    'turnover' : True,
    'pct_chg_stk' : True
}


# 回测基础设置
start_date = '20220801' # 开始日期
end_date = '20240517' # 结束日期
hold_num = 10 # 持有数量
threadhold_num = 0 # 轮动阈值
c_rate =  2 / 1000 # 买卖一次花费的总佣金和滑点（双边）
benchmark = 'index_jsl' # 选择基准，集思录等权:index_jsl, 沪深300:index_300, 中证1000:index_1000, 国证2000:index_2000
shares_per_board_lot = 10 # 每手数量(最小交易单位)

SP = 0.06 # 盘中止盈条件

Riskfree_rate = 0.023 # 无风险收益率

#读取禄得数据
cbdf = pd.read_parquet('cb_data.pq') # 导入转债数据
index = pd.read_parquet('index.pq') # 导入指数数据

#先计算自定义指标-溢价率偏离率
cbdf['conv_bias15'] = cbdf.groupby('code')['conv_prem'].transform(lambda x: x - x.shift(1).rolling(15).mean())
#再计算自定义指标-五日涨跌幅差,公式为cbdf中按照code分组，pct_chg_5列的值-pct_chg_5_stk列的值
cbdf['pct_chg_gap5'] = cbdf.groupby('code').apply(lambda x: x['pct_chg_5'] - x['pct_chg_5_stk']).reset_index(level=0, drop=True)

'''EMA指标暂时不用，因为回测中没有用到
#再计算溢价率的EMA指标
cbdf['conv_prem_ema'] = cbdf.groupby('code')['conv_prem'].transform(lambda x: talib.EMA(x, timeperiod=15).tolist())
#再计算当日溢价率与前一日的溢价率EMA的差值

# 对每组（每只股票）的conv_prem_ema列应用.shift()，以获取前一天的EMA值
cbdf['prev_day_conv_prem_ema'] = cbdf.groupby('code')['conv_prem_ema'].shift(1)

# 计算每日conv_prem与前一天conv_prem_ema的差值
cbdf['conv_prem_diff'] = cbdf['conv_prem'] - cbdf['prev_day_conv_prem_ema']
'''

#选取回测日期内数据
cbdf = cbdf[(cbdf.index.get_level_values('trade_date') >= start_date) & (cbdf.index.get_level_values('trade_date') <= end_date)] # 选择时间范围内数据

    # 排除设置

cbdf['filter'] = False # 初始化过滤器

cbdf['close_pct'] = cbdf.groupby('trade_date')['close'].rank(pct=True) # 将收盘从小到大百分比排列

cbdf.loc[cbdf.is_call.isin(['已公告强赎', '公告到期赎回','公告实施强赎', '公告提示强赎', '已满足强赎条件']), 'filter'] = True # 排除赎回状态
cbdf.loc[cbdf.list_days <= 3, 'filter'] = True # 排除新债
cbdf.loc[cbdf.left_years < 0.5, 'filter'] = True # 排除到期日小于0.5年的标的
cbdf.loc[cbdf.left_years > 5, 'filter'] = True # 排除到期日大于5年的标的

cbdf.loc[cbdf.conv_prem >0.5 , 'filter'] = True # 排除溢价率
cbdf.loc[cbdf.remain_cap > 5, 'filter'] = True # 排除剩余市值
#cbdf.loc[cbdf.dif < 0, 'filter'] = True # 排除DIF

cbdf.loc[cbdf.name_stk.str.upper().str.contains('ST'), 'filter'] = True # 排除ST
cbdf.loc[cbdf.name_stk.str.upper().str.contains('退'), 'filter'] = True # 排除退市

cbdf.loc[cbdf.rating.isin(['BBB+', 'BBB','BBB+', 'BB+','BB', 'BB-','B+', 'B','B-', 'CCC+','CCC', 'CCC-']), 'filter'] = True # 排除评级
cbdf.loc[cbdf.close >140, 'filter'] = True # 排除价格


def simulate_returns(selected_factors, weights,df):
    
    #深拷贝df，避免影响全局cbdf
    df = copy.deepcopy(df)
    # 生成因子字典，name:列名，weight:权重, ascending:排序方向

    # 将selected_factors, weights展开赋值给rank_factors
    rank_factors = [{'name': factor, 'weight': weight} for factor, weight in zip(selected_factors, weights)]
    #将rank_factors中的ascending列赋值为factor_direction中factor_name对应的布尔值
    for factor in rank_factors:
        factor['ascending'] = factor_direction[factor['name']]


    #df.loc[df.ps_ttm < 0, 'filter'] = True # 排除市盈率
    #df.loc[df.ps_ttm < 0, 'filter'] = True # 排除市销率
    #df.loc[df.pb < 0, 'filter'] = True # 排除市净率
    #df.loc[df.close_pct > 0.8, 'filter'] = True # 排除收盘价高于x%的标的


    # 计算多因子得分 和 排名(score总分越大越好， rank总排名越小越好)

    # 生成因子字典，name:列名，weight:权重, ascending:排序方向
    #rank_factors = [
    #    {'name': 'conv_bias15', 'weight': 3, 'ascending': False}, 
    #    {'name': 'pct_chg_gap5', 'weight': 1, 'ascending': False}, ]

    # 计算多因子得分 和 排名(score总分越大越好， rank总排名越小越好)
    trade_date_group = df[df['filter'] ==False].groupby('trade_date')
    for factor in rank_factors:
        if factor['name'] in df.columns:
            df[f'{factor["name"]}_score'] = trade_date_group[factor["name"]].rank(ascending=factor['ascending']) * factor['weight']
        else:
            print(f'未找到因子【{factor["name"]}】, 跳过')

    df['score'] = df[df.filter(like='score').columns].sum(axis=1, min_count=1)

    if hold_num >= 1:
        df['rank'] = df.groupby('trade_date')['score'].rank('first', ascending=False)
    else:
        df['rank_pct'] = df.groupby('trade_date')['score'].rank('first', ascending=False, pct=False)

    # 处理止盈

    code_group = df.groupby('code')
    # (2)次日止盈条件
    df['aft_open'] = code_group.open.shift(-1) # 计算次日开盘价
    df['aft_close'] = code_group.close.shift(-1) # 计算次日收盘价
    df['aft_high'] = code_group.high.shift(-1) # 计算次日最高价
    df['time_return']= code_group.pct_chg.shift(-1) # 先计算不止盈情况的收益率
    df['SFZY']='未满足止盈' #先记录默认情况
    pd.set_option('display.max_columns', None)  # 当列太多时不换行

    df.loc[df['aft_high'] >= df['close'] * (1+SP),'time_return'] = SP # 满足止盈条件止盈
    df.loc[df['aft_open'] >= df['close'] * (1+SP),'time_return'] = \
    (df['aft_open']-df['close'])/df['close'] # 开盘满足止盈条件则按开盘价计算涨幅
    df.loc[df['aft_high'] >= df['close'] * (1+SP),'SFZY'] = '满足止盈'


    # 计算每日信号 采样信号 持仓状态，time_return已经在上一步止盈时计算
    code_group = df.groupby('code')
    df.loc[(df['rank'] <= hold_num), 'signal'] = 1 # 标记信号
    df.dropna(subset=['signal'], inplace=True) # 删除没有标记的行
    df.sort_values(by='trade_date', inplace=True) # 按日期排序


    # 计算组合回报
    res = pd.DataFrame()
    res['time_return'] = df.groupby('trade_date')['time_return'].mean() # 按等权计算组合回报
    # 计算手续费
    pos_df = df['signal'].unstack('code')
    pos_df.fillna(0, inplace=True)
    res['cost'] = pos_df.diff().abs().sum(axis=1) * c_rate / (pos_df.shift().sum(axis=1) + pos_df.sum(axis=1))
    res.iloc[0, 1] = 0.5 * c_rate# 修正首行手续费
    res['time_return'] = (res['time_return'] + 1) * (1 - res['cost']) - 1# 扣除手续费及佣金后的回报

    #res.to_excel('res.xlsx')

    #计算净值(区间总回报)
    res = res[:-1]
    cumulative_returns = (1 + res['time_return']).cumprod()
    total_return = cumulative_returns.iloc[-1]
    #print('区间总回报：',total_return)

    # 计算回测所用的时间
    end_time = datetime.now()

    time_used = end_time - start_time
    # 把time_used格式化为时分秒
    time_used = str(timedelta(seconds=int(time_used.total_seconds())))

    #print('回测所用时间：', time_used)

    # 生成回测文件

    # 对`date`列分组，并聚合`name`列
    df.reset_index(inplace=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    # 按照'trade_date'和'rank'列对df进行排序
    # 假设rank已经是正确的数据类型，如果不是，需要先进行转换
    df.sort_values(by=['trade_date', 'rank'], ascending=[True, True], inplace=True)
    # 使用自定义的聚合函数来聚合排序后的'names'
    def sorted_names(series):
        return series.tolist()
    # 对排序后的df进行分组和聚合

    df_grouped = df.groupby(df['trade_date'].dt.date)['name'].agg(sorted_names).reset_index()

    # 重命名列以便清晰理解
    df_grouped.rename(columns={'trade_date':'日期','name': '持仓'}, inplace=True)
    #df_grouped.to_excel('CONVIAS2.0回测持仓.xlsx', index=False)

    # 生成回测报告
  
    cagr = qs.stats.cagr(res.time_return)
    sharpe = qs.stats.sharpe(res.time_return, rf=Riskfree_rate,periods=242)

    #print CAGR和Sharpe,CAGR为2位小数的百分数，Sharpe为3位小数的浮点数
    #print(f"CAGR: {cagr:.2%}, Sharpe: {sharpe:.3f}")
    
    return cagr , sharpe


def objective(trial):

    # 将factors转换位字符串避免Optuna分布式警告,使用前转换回元组

    # 将元组转化为字符串形式，然后再放入列表中
    factor_combinations = ['*'.join(combo) for combo in itertools.combinations(factors.keys(), 3)]
    selected_factors = trial.suggest_categorical('selected_factors', factor_combinations)

    # 将选中的因子从字符串转化回你需要的元组形式
    selected_factors = tuple(selected_factors.split('*'))

    # 为了确保至少有一个非0权重，将第一个权重设置为1到5之间
    weights = [trial.suggest_int(f'weight_0', 1, 5)]
    # 其余的权重则可以为0到5之间
    weights.extend(trial.suggest_int(f'weight_{i}', 0, 5) for i in range(1, len(selected_factors)))

    
    # 计算回报
    cagr ,sharpe= simulate_returns(selected_factors, weights,cbdf)

    return cagr,sharpe

def optimize(n_trials):
    study = optuna.create_study(directions=['maximize', 'maximize'])
    study.optimize(objective, n_trials=n_trials)


    # 获取所有最佳试验的列表
    best_trials = study.best_trials
    
    # 获取每个最佳试验的所有目标值
    best_values = [[trial.values[i] for trial in best_trials] for i in range(len(best_trials[0].values))]
    
    # 获取每个最佳试验的参数
    best_params = [trial.params for trial in best_trials]
    
    # 返回每个目标的最佳值列表和最佳参数列表
    return best_values, best_params
    


if __name__ == '__main__':

    start_time = datetime.now()

    # 确定并行的进程数，通常设置为CPU核心数
    n_processes = 36
    n_trials_per_process = 2739  # 每个进程的试验次数
    # 使用Joblib's Parallel实现并行计算
    results  = Parallel(n_jobs=n_processes)(
        delayed(optimize)(n_trials_per_process) for _ in range(n_processes))
    

    # 解包所有的最佳值和最佳参数
    best_values, best_params_list = zip(*results)
    # 找到最大的最佳值，并获取对应的参数
    max_value_index = best_values.index(max(best_values))
    best_parameters = best_params_list[max_value_index]
    #将max_value_index和best_parameters写入'report.xlsx'文件
    
    # 将 best_params_list 转换为 DataFrame
    params_df = pd.DataFrame(best_params_list)

    # 将 best_values 添加为一个新列到 params_df
    params_df['Best Value'] = best_values

    # 确保 DataFrame 的索引列不是默认的数字索引
    params_df.reset_index(drop=True, inplace=True)

    # 将 DataFrame 写入 Excel 文件
    params_df.to_excel('report.xlsx', index=False, header=True, engine='openpyxl')


    # 输出最好的值和参数
    print("最佳值:", max(best_values))
    print("对应的最佳参数:", best_parameters)

    # 计算回测所用的时间
    end_time = datetime.now()

    time_used = end_time - start_time
    # 把time_used格式化为时分秒
    time_used = str(timedelta(seconds=int(time_used.total_seconds())))

    print('回测所用时间：', time_used)


