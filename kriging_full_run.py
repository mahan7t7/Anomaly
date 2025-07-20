import os
import time
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ⏱ شروع زمان‌سنجی
start_time = time.time()

# 📂 مسیر خروجی
OUTPUT_DIR = "outputs/tile_kriging"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📥 بارگذاری داده‌ها
df = pd.read_csv("iran_elev_gravity.csv")

# 🎯 تبدیل فقط ستون‌های عددی به float32
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].astype(np.float32)

# 🎯 تنظیمات tile
tile_size = 5000      # تعداد نقاط آموزش در هر tile
test_size = 1000      # تعداد نقاط تست در هر tile
max_tiles = int(len(df) / tile_size)

results = []

for i in range(max_tiles):
    print(f"🧩 اجرای tile {i+1}/{max_tiles}...")

    start_idx = i * tile_size
    end_idx = min(start_idx + tile_size, len(df))
    tile_df = df.iloc[start_idx:end_idx]

    if len(tile_df) < test_size + 50:
        print("⚠️ داده‌های ناکافی در tile. رد شد.")
        continue

    train_df = tile_df.sample(n=tile_size - test_size, random_state=42)
    test_df = tile_df.drop(train_df.index).sample(n=test_size, random_state=42)

    x_train = train_df['lon'].values
    y_train = train_df['lat'].values
    z_train = train_df['gravity_anomaly'].values

    x_test = test_df['lon'].values
    y_test = test_df['lat'].values
    z_test = test_df['gravity_anomaly'].values

    try:
        OK = OrdinaryKriging(
            x_train, y_train, z_train,
            variogram_model='spherical',
            verbose=False,
            enable_plotting=False
        )

        z_pred = []
        for lon, lat in zip(x_test, y_test):
            z, _ = OK.execute('points', [lon], [lat])
            z_pred.append(z[0])

        mae = mean_absolute_error(z_test, z_pred)
        rmse = np.sqrt(mean_squared_error(z_test, z_pred))
        r2 = r2_score(z_test, z_pred)

        results.append({
            'tile': i + 1,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'train_size': len(train_df),
            'test_size': len(test_df)
        })

        df_out = pd.DataFrame({
            'tile': i + 1,
            'lon': x_test,
            'lat': y_test,
            'true_anomaly': z_test,
            'predicted_anomaly': z_pred
        })
        df_out.to_csv(os.path.join(OUTPUT_DIR, f"tile_{i+1}_predictions.csv"), index=False)

    except Exception as e:
        print(f"❌ خطا در tile {i+1}: {str(e)}")
        continue

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "tilewise_kriging_results.csv"), index=False)

print("✅ همه tile‌ها پردازش شدند.")
print(f"⏱ کل زمان اجرا: {(time.time() - start_time)/60:.2f} دقیقه")
