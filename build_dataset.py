import pandas as pd
import glob
import os
import numpy as np

# ================= 配置项 =================
# 数据文件夹路径
DATA_DIR = './data'
# 时间戳格式的 CSV 文件模式（自动扫描所有匹配的文件）
TIMESTAMP_FILE_PATTERN = '*-h1-bid-*.csv'
# 想要生成的文件名
OUTPUT_FILE = './data/G31_RawPrice.txt'
# =========================================

def extract_pair_name(filename):
    """
    从文件名提取货币对名称
    例如: 'xauusd-h1-bid-2020-01-01-2024-12-31.csv' -> 'XAUUSD'
         'eurusd-h1-bid-2020-01-01-2024-12-31.csv' -> 'EURUSD'
    """
    # 去掉扩展名，取第一部分（在第一个 '-' 之前）
    base_name = os.path.splitext(filename)[0]
    pair_name = base_name.split('-')[0].upper()
    return pair_name

def main():
    # 用于存放所有货币对数据的列表
    series_list = []
    
    # ========== 处理 data 文件夹中的所有时间戳格式文件 ==========
    print("="*50)
    print("📅 处理 data 文件夹中的所有时间戳格式文件...")
    print("="*50)
    
    # 自动扫描所有匹配的 CSV 文件
    csv_files = glob.glob(os.path.join(DATA_DIR, TIMESTAMP_FILE_PATTERN))
    
    if len(csv_files) == 0:
        print(f"❌ 错误：在 {DATA_DIR} 文件夹中没有找到匹配 '{TIMESTAMP_FILE_PATTERN}' 的文件！")
        return
    
    print(f"🔍 找到了 {len(csv_files)} 个时间戳格式的数据文件，准备开始处理...\n")
    
    for file_path in sorted(csv_files):
        filename = os.path.basename(file_path)
        # 从文件名提取货币对名称
        pair_name = extract_pair_name(filename)
        
        print(f"   -> 正在读取: {pair_name} ({filename}) ...")
        
        try:
            # 读取 CSV
            df = pd.read_csv(file_path)
            print(f"      📊 原始数据形状: {df.shape}")
            
            # 检查是否有 timestamp 列
            if 'timestamp' not in df.columns:
                print(f"      ⚠️ 警告：文件没有 'timestamp' 列，跳过")
                continue
            
            # 将毫秒级时间戳转换为 datetime
            # 时间戳是毫秒级（13位），需要除以1000转换为秒级
            df['time'] = pd.to_datetime(df['timestamp'] / 1000, unit='s')
            df.set_index('time', inplace=True)
            
            # 检查是否有 close 列
            if 'close' not in df.columns:
                print(f"      ⚠️ 警告：文件没有 'close' 列，跳过")
                continue
            
            # 只取 'close' 列（小写），并重命名为对应的名称
            close_series = df[['close']].rename(columns={'close': pair_name})
            
            # 去除重复的时间索引 (以防万一)
            close_series = close_series[~close_series.index.duplicated(keep='first')]
            
            series_list.append(close_series)
            print(f"      ✅ 成功读取 {len(close_series)} 行数据，时间范围: {close_series.index.min()} 到 {close_series.index.max()}")
            
        except Exception as e:
            print(f"      ❌ 读取 {filename} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n   ✅ 已处理 {len(series_list)} 个时间戳格式的文件\n")

    # ========== 第二步：合并所有数据 ==========
    print("="*50)
    print("🔗 第二步：合并所有数据...")
    print("="*50)
    
    # 使用 outer join 确保并集，保证时间轴是完整的
    print("⏳ 正在按时间轴对齐合并...")
    final_df = pd.concat(series_list, axis=1)
    
    # 按时间排序
    final_df.sort_index(inplace=True)
    
    print(f"   原始数据形状: {final_df.shape} (行=时间, 列=货币对)")
    
    # 3. 处理缺失值 (Data Cleaning)
    # 外汇数据因为周末休市，大家应该都是空的，可以drop
    # 或者某些时刻个别货币缺失，用 ffill (前向填充)
    
    # 策略：先用前向填充(fill forward)补全偶尔的交易缺失
    final_df.ffill(inplace=True)
    # 再用后向填充(back fill)补全开头可能的缺失
    final_df.bfill(inplace=True)
    
    # 如果还有整行都是空的(比如周末)，直接丢弃
    original_len = len(final_df)
    final_df.dropna(axis=0, how='any', inplace=True)
    print(f"   去除包含 NaN 的行后形状: {final_df.shape} (删除了 {original_len - len(final_df)} 行)")

    # 4. 保存为 txt 文件
    # header=False (不保存列名), index=False (不保存时间列)
    # MTGNN 只要纯数字矩阵
    final_df.to_csv(OUTPUT_FILE, sep=',', header=False, index=False)
    
    print("\n" + "="*50)
    print(f"✅ 处理完成！")
    print(f"📂 输出文件已保存为: {OUTPUT_FILE}")
    print(f"📊 最终矩阵大小: {final_df.shape}")
    print(f"   (行数应当作为 seq_in_len 的参考，列数应为 {final_df.shape[1]})")
    print(f"   (包含 {len(series_list)} 个时间戳格式文件 = {final_df.shape[1]} 列)")
    print("="*50)
    
    # 简单检查一下生成的数据
    # print(final_df.head())

if __name__ == "__main__":
    main()