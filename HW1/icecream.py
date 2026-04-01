import pandas as pd
import matplotlib.pyplot as plt

# 1. 讀取數據 (請確認檔案路徑)
df = pd.read_csv('IceCreamData.csv') 

# 2. 基本描述與統計
print("Dataset Statistics:")
print(df.describe()) # 這會直接輸出 mean, std, min, max

# 3. 繪製散佈圖
plt.figure(figsize=(8, 6))
plt.scatter(df['Temperature'], df['Revenue'], color='blue', s=10, alpha=0.5)
plt.title('Ice Cream Revenue vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Revenue ($)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()