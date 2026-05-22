import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


# ── LOAD ────────────────────────────────────────────────────────
df = pd.read_csv(
    "C:/Users/suraj/OneDrive/Desktop/Projects/houseprice_pridiction/Kolkata.csv"
)


# ── CLEAN ───────────────────────────────────────────────────────
df = df[df['Price'] > 0]
df = df[df['Area']  > 0]
df.dropna(subset=['Price', 'Area', 'Location', 'No. of Bedrooms'], inplace=True)

# Remove bad bedroom count
df = df[df['No. of Bedrooms'] != 9]

# Remove rows with impossible Price/sqft (main data quality fix)
df['PricePerSqft'] = df['Price'] / df['Area']
df = df[(df['PricePerSqft'] >= 2000) & (df['PricePerSqft'] <= 25000)]
df = df.drop(columns='PricePerSqft')

# Remove Area outliers
low, high = df['Area'].quantile(0.01), df['Area'].quantile(0.99)
df = df[(df['Area'] >= low) & (df['Area'] <= high)]

df = df.reset_index(drop=True)
print(f"Rows after cleaning : {len(df)}")


# ── FEATURE ENGINEERING ─────────────────────────────────────────
binary_cols = [
    'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens',
    'JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall',
    'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security',
    'PowerBackup', 'CarParking', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom',
    'Hospital', 'WashingMachine', 'Gasconnection', 'AC', 'Wifi',
    "Children'splayarea", 'LiftAvailable', 'BED', 'VaastuCompliant',
    'Microwave', 'GolfCourse', 'TV', 'DiningTable', 'Sofa', 'Wardrobe',
    'Refrigerator', 'Resale'
]
df[binary_cols] = df[binary_cols].fillna(0)

df['AmenityScore']   = df[binary_cols].sum(axis=1)
df['AreaPerBedroom'] = df['Area'] / df['No. of Bedrooms'].replace(0, 1)


# ── FEATURES & TARGET ───────────────────────────────────────────
feature_cols = (
    ['Area', 'Location', 'No. of Bedrooms', 'AreaPerBedroom', 'AmenityScore']
    + binary_cols
)

X = df[feature_cols].copy()
Y = df['Price']


# ── TARGET ENCODE LOCATION ──────────────────────────────────────
def target_encode(train_df, test_df, col='Location', target='Price'):
    median_map    = train_df.groupby(col)[target].median()
    global_median = train_df[target].median()
    train_out, test_out = train_df.copy(), test_df.copy()
    train_out[col] = train_out[col].map(median_map).fillna(global_median)
    test_out[col]  = test_out[col].map(median_map).fillna(global_median)
    return train_out, test_out


# ── TRAIN / TEST SPLIT ──────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

tr = X_train.copy(); tr['Price'] = Y_train.values
te = X_test.copy();  te['Price'] = Y_test.values
tr_enc, te_enc  = target_encode(tr, te)
X_train_enc     = tr_enc.drop(columns='Price')
X_test_enc      = te_enc.drop(columns='Price')


# ── CROSS-VALIDATION ────────────────────────────────────────────
kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_enc)):
    X_tr,  X_val = X_train_enc.iloc[tr_idx].copy(), X_train_enc.iloc[val_idx].copy()
    Y_tr,  Y_val = Y_train.iloc[tr_idx],             Y_train.iloc[val_idx]

    f_tr  = X_tr.copy();  f_tr['Price']  = Y_tr.values
    f_val = X_val.copy(); f_val['Price'] = Y_val.values
    f_tr_enc, f_val_enc = target_encode(f_tr, f_val)

    m = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=5, subsample=0.8, random_state=42
    )
    m.fit(f_tr_enc.drop(columns='Price'), Y_tr)
    score = r2_score(Y_val, m.predict(f_val_enc.drop(columns='Price')))
    cv_r2.append(score)
    print(f"  Fold {fold+1} R²: {score:.4f}")

print(f"\nMean CV R²        : {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")


# ── TRAIN FINAL MODEL ───────────────────────────────────────────
model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)
model.fit(X_train_enc, Y_train)

print(f"Training R² Score : {r2_score(Y_train, model.predict(X_train_enc)):.4f}")
print(f"Test R²  Score    : {r2_score(Y_test,  model.predict(X_test_enc)):.4f}")


# ── FEATURE IMPORTANCE ──────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=X_train_enc.columns)
print("\nTop 10 Features:")
print(importances.sort_values(ascending=False).head(10).round(4))


# ── SAVE ────────────────────────────────────────────────────────
joblib.dump({
    'model'           : model,
    'location_medians': df.groupby('Location')['Price'].median(),
    'global_median'   : Y.median(),
    'feature_cols'    : feature_cols,
    'binary_cols'     : binary_cols
}, "house_price_model.pkl")

print("\nModel saved as house_price_model.pkl")