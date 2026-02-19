import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

inputs = []
outputs = []

for f in glob.glob("train/input_*.csv"):
    inputs.append(pd.read_csv(f))

for f in glob.glob("train/output_*.csv"):
    outputs.append(pd.read_csv(f))

input_df = pd.concat(inputs, ignore_index=True)
output_df = pd.concat(outputs, ignore_index=True)

input_df = input_df.sort_values(["game_id","play_id","nfl_id","frame_id"])

for lag in [1,2,3,5]:
    input_df[f"x_lag{lag}"] = input_df.groupby(["game_id","play_id","nfl_id"])["x"].shift(lag)
    input_df[f"y_lag{lag}"] = input_df.groupby(["game_id","play_id","nfl_id"])["y"].shift(lag)
    input_df[f"s_lag{lag}"] = input_df.groupby(["game_id","play_id","nfl_id"])["s"].shift(lag)

key = ["game_id","play_id","nfl_id"]

last_input = input_df.loc[input_df.groupby(key)["frame_id"].idxmax()]

df = output_df.merge(last_input, on=key, suffixes=("_target","_input"))
df["k"] = df["frame_id_target"]

df["dt"] = df["num_frames_output"]
df["dx"] = df["x_target"] - df["x_input"]
df["dy"] = df["y_target"] - df["y_input"]
df["dir_rad"] = np.deg2rad(df["dir"])
df["vx"] = df["s"] * np.cos(df["dir_rad"])
df["vy"] = df["s"] * np.sin(df["dir_rad"])
df["k2"] = df["k"]**2
df["vx_k"] = df["vx"] * df["k"]
df["vy_k"] = df["vy"] * df["k"]
df["a_k2"] = df["a"] * df["k"]**2
df["o_rad"] = np.deg2rad(df["o"])
df["ox"] = np.cos(df["o_rad"])
df["oy"] = np.sin(df["o_rad"])

df["dist_ball_x"] = df["ball_land_x"] - df["x_input"]
df["dist_ball_y"] = df["ball_land_y"] - df["y_input"]
df["dist_ball"] = np.sqrt(df["dist_ball_x"]**2 + df["dist_ball_y"]**2)
df["angle_ball"] = np.arctan2(df["dist_ball_y"], df["dist_ball_x"])
df["dot_ball_vel"] = (df["dist_ball_x"] * df["vx"] + df["dist_ball_y"] * df["vy"])

cat_cols = ["player_position", "player_role", "player_side", "play_direction"]

def height_to_inches(h):
    try:
        feet, inch = h.split("-")
        return int(feet)*12 + int(inch)
    except:
        return np.nan

df["player_height_in"] = df["player_height"].apply(height_to_inches)
df["player_height_in"] = df["player_height_in"].fillna(df["player_height_in"].median())

for c in cat_cols:
    df[c] = df[c].astype("category").cat.codes

features = [
    "x_input","y_input",
    "vx","vy","a","k","k2","vx_k","vy_k","a_k2",
    "o_rad","ox","oy",
    "player_weight",
    "player_height_in",
    "player_position",
    "player_role",
    "player_side",
    "play_direction",
    "x_lag1","x_lag2","x_lag3","x_lag5",
    "y_lag1","y_lag2","y_lag3","y_lag5",
    "s_lag1","s_lag2","s_lag3","s_lag5",
    "ball_land_x",
    "ball_land_y",
    "dist_ball_x",
    "dist_ball_y",
    "dist_ball",
    "angle_ball",
    "dot_ball_vel"
]

df["dx_n"] = df["dx"] / df["k"]
df["dy_n"] = df["dy"] / df["k"]

y = df[["dx_n","dy_n"]]
groups = df["game_id"].astype(str) + "_" + df["play_id"].astype(str)

left = df["play_direction"] == "left"
df.loc[left, "ball_land_x"] = 120 - df.loc[left, "ball_land_x"]

for col in ["x_input","x_lag1","x_lag2","x_lag3","x_lag5"]:
    df.loc[left, col] = 120 - df.loc[left, col]

df.loc[left, "vx"] *= -1

X = df[features]

def metric(y_true, y_pred):
    return np.sqrt(
        np.mean(
            (y_true[:,0]-y_pred[:,0])**2 +
            (y_true[:,1]-y_pred[:,1])**2
        ) / 2
    )

gkf = GroupKFold(n_splits=5)
scores = []

for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr_x = y.iloc[tr, 0]
    y_tr_y = y.iloc[tr, 1]
    y_va_x = y.iloc[va, 0]
    y_va_y = y.iloc[va, 1]

    model_x = XGBRegressor(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=50
    )

    model_y = XGBRegressor(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=50
    )

    model_x.fit(X_tr, y_tr_x, eval_set=[(X_va, y_va_x)], verbose=False)
    model_y.fit(X_tr, y_tr_y, eval_set=[(X_va, y_va_y)], verbose=False)

    pred_x = model_x.predict(X_va)
    pred_y = model_y.predict(X_va)
    pred = np.column_stack([pred_x, pred_y])

    k_val = df.iloc[va]["k"].values.reshape(-1,1)
    pred_real = pred * k_val
    y_real = df.iloc[va][["dx","dy"]].values

    sc = metric(y_real, pred_real)
    scores.append(sc)
    print(f"Fold {fold}: {sc:.4f}")

print("CV mean:", np.mean(scores))
