"""
Data Preprocessing Pipeline for MSIS 522 HW1
Flaredown Food & Flare Prediction

Loads the Flaredown export CSV, filters to user-days with both food and symptom
tracking, engineers features (food pivots, treatment pivots, weather, tags,
demographics), defines the binary flare target, and saves processed data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import gc
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CSV_PATH = Path(r"C:\Users\hanna\.cache\kagglehub\datasets\flaredown\flaredown-autoimmune-symptom-tracker\versions\2\export.csv")

# Food category mappings
FOOD_CATEGORIES = {
    'dairy': ['cheese', 'milk', 'yogurt', 'butter', 'cream', 'ice cream', 'whey',
              'cream cheese', 'sour cream', 'cottage cheese', 'mozzarella', 'cheddar'],
    'grains': ['bread', 'rice', 'pasta', 'wheat', 'oats', 'cereal', 'flour',
               'crackers', 'tortilla', 'bagel', 'noodles', 'corn', 'quinoa',
               'granola', 'oatmeal', 'toast', 'sandwich'],
    'fruits': ['apple', 'banana', 'berries', 'strawberry', 'blueberry', 'orange',
               'grapes', 'watermelon', 'pineapple', 'mango', 'peach', 'lemon',
               'avocado', 'tomato', 'fruit'],
    'protein': ['chicken', 'beef', 'pork', 'fish', 'egg', 'eggs', 'turkey', 'bacon',
                'sausage', 'steak', 'salmon', 'shrimp', 'tuna', 'ham', 'meat',
                'lamb', 'protein'],
    'caffeine': ['coffee', 'tea', 'caffeine', 'espresso', 'energy drink', 'green tea',
                 'black tea', 'matcha', 'soda', 'cola', 'coke', 'pepsi'],
    'sugar': ['sugar', 'chocolate', 'candy', 'cookie', 'cake', 'dessert', 'ice cream',
              'sweets', 'honey', 'syrup', 'donut', 'pie', 'brownie', 'pastry']
}


def load_raw_data():
    """Load the raw Flaredown CSV."""
    print(f"Loading raw data from {CSV_PATH}...")
    df = pd.read_csv(
        CSV_PATH,
        dtype={
            'user_id': str,
            'sex': str,
            'country': str,
            'trackable_type': str,
            'trackable_name': str,
            'trackable_value': str,
        },
        parse_dates=['checkin_date'],
        low_memory=False
    )
    print(f"  Raw data: {len(df):,} rows, {df.shape[1]} columns")
    print(f"  Trackable types: {df['trackable_type'].value_counts().to_dict()}")
    return df


def filter_food_and_symptom_days(df):
    """Keep only user-days where the user tracked BOTH food AND symptoms."""
    food_ud = df.loc[df['trackable_type'] == 'Food', ['user_id', 'checkin_date']].drop_duplicates()
    symp_ud = df.loc[df['trackable_type'] == 'Symptom', ['user_id', 'checkin_date']].drop_duplicates()
    valid_days = food_ud.merge(symp_ud, on=['user_id', 'checkin_date'])
    print(f"  User-days with both food & symptoms: {len(valid_days):,}")

    df_filtered = df.merge(valid_days, on=['user_id', 'checkin_date'])
    print(f"  Filtered rows: {len(df_filtered):,}")
    return df_filtered, valid_days


def create_target(df_filtered, valid_days):
    """Binary flare target: max symptom severity >= 3 on that day."""
    symptoms = df_filtered[df_filtered['trackable_type'] == 'Symptom'].copy()
    symptoms['val'] = pd.to_numeric(symptoms['trackable_value'], errors='coerce')

    max_sev = symptoms.groupby(['user_id', 'checkin_date'], observed=True)['val'].max().reset_index()
    max_sev.columns = ['user_id', 'checkin_date', 'max_symptom_severity']
    max_sev['flare'] = (max_sev['max_symptom_severity'] >= 3).astype(int)

    target_df = valid_days.merge(max_sev, on=['user_id', 'checkin_date'], how='left')
    target_df['flare'] = target_df['flare'].fillna(0).astype(int)
    print(f"  Flare distribution: {target_df['flare'].value_counts().to_dict()}")
    return target_df


def clean_food_name(name):
    if not isinstance(name, str):
        return name
    name = name.lower().strip()
    for suffix in [' (organic)', ' (raw)', ' (cooked)', ' (fresh)', ' (frozen)']:
        name = name.replace(suffix, '')
    return name


def add_binary_feature(result_df, subset_df, name_val, col_name):
    """Add a single binary column to result_df from subset matching name_val."""
    rows = subset_df[subset_df['trackable_name'] == name_val][['user_id', 'checkin_date']].drop_duplicates()
    rows[col_name] = np.int8(1)
    result_df = result_df.merge(rows, on=['user_id', 'checkin_date'], how='left')
    result_df[col_name] = result_df[col_name].fillna(0).astype(np.int8)
    return result_df


def pivot_foods(df_filtered, target_df, top_n=50):
    """Pivot top N foods into binary columns + category rollups."""
    foods = df_filtered[df_filtered['trackable_type'] == 'Food'].copy()
    foods['trackable_name'] = foods['trackable_name'].apply(clean_food_name)

    top_foods = foods['trackable_name'].value_counts().head(top_n).index.tolist()
    print(f"  Top {top_n} foods (most common: {top_foods[:5]})")

    result = target_df.copy()
    food_cols = []
    for i, food in enumerate(top_foods):
        col = f"food_{food.replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '')}"
        result = add_binary_feature(result, foods, food, col)
        food_cols.append(col)
        if (i + 1) % 10 == 0:
            print(f"    ...processed {i + 1}/{top_n} foods")

    # Category rollups using string matching on concatenated foods per day
    user_day_foods = foods.groupby(['user_id', 'checkin_date'], observed=True)['trackable_name'].apply(
        lambda x: ' | '.join(x.values)
    ).reset_index()
    user_day_foods.columns = ['user_id', 'checkin_date', '_foods_str']
    result = result.merge(user_day_foods, on=['user_id', 'checkin_date'], how='left')
    result['_foods_str'] = result['_foods_str'].fillna('')

    for cat_name, keywords in FOOD_CATEGORIES.items():
        col = f"foodcat_{cat_name}"
        pattern = '|'.join(keywords)
        result[col] = result['_foods_str'].str.contains(pattern, case=False, na=False).astype(np.int8)

    result = result.drop(columns=['_foods_str'])
    print(f"  Food features: {len(food_cols)} individual + {len(FOOD_CATEGORIES)} categories")
    return result, food_cols


def pivot_treatments(df_filtered, result_df, top_n=20):
    """Pivot top N treatments into binary columns."""
    treats = df_filtered[df_filtered['trackable_type'] == 'Treatment'].copy()
    if len(treats) == 0:
        print("  No treatment data, skipping.")
        return result_df, []

    treats['trackable_name'] = treats['trackable_name'].str.lower().str.strip()
    top = treats['trackable_name'].value_counts().head(top_n).index.tolist()

    treat_cols = []
    for tn in top:
        col = f"treat_{tn.replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '')}"
        result_df = add_binary_feature(result_df, treats, tn, col)
        treat_cols.append(col)

    print(f"  Treatment features: {len(treat_cols)}")
    return result_df, treat_cols


def pivot_weather(df_filtered, result_df):
    """Pivot weather data into numeric columns."""
    weather = df_filtered[df_filtered['trackable_type'] == 'Weather'].copy()
    if len(weather) == 0:
        print("  No weather data, skipping.")
        return result_df, []

    weather_cols = []

    # Numeric weather
    for wn in ['humidity', 'precipIntensity', 'pressure', 'temperatureMin', 'temperatureMax']:
        sub = weather[weather['trackable_name'] == wn].copy()
        if len(sub) == 0:
            continue
        sub['val'] = pd.to_numeric(sub['trackable_value'], errors='coerce')
        agg = sub.groupby(['user_id', 'checkin_date'], observed=True)['val'].mean().reset_index()
        col = f"weather_{wn}"
        agg.columns = ['user_id', 'checkin_date', col]
        result_df = result_df.merge(agg, on=['user_id', 'checkin_date'], how='left')
        result_df[col] = result_df[col].fillna(result_df[col].median())
        weather_cols.append(col)

    # Weather icon
    icon_data = weather[weather['trackable_name'] == 'icon'].drop_duplicates(
        subset=['user_id', 'checkin_date']
    )[['user_id', 'checkin_date', 'trackable_value']].copy()
    if len(icon_data) > 0:
        icon_data.columns = ['user_id', 'checkin_date', '_icon']
        result_df = result_df.merge(icon_data, on=['user_id', 'checkin_date'], how='left')
        top_icons = result_df['_icon'].value_counts().head(8).index.tolist()
        for icon in top_icons:
            col = f"weather_icon_{icon.replace('-', '_').replace(' ', '_')}"
            result_df[col] = (result_df['_icon'] == icon).astype(np.int8)
            weather_cols.append(col)
        result_df = result_df.drop(columns=['_icon'])

    print(f"  Weather features: {len(weather_cols)}")
    return result_df, weather_cols


def pivot_tags(df_filtered, result_df):
    """Pivot tag data into binary columns."""
    tags = df_filtered[df_filtered['trackable_type'] == 'Tag'].copy()
    if len(tags) == 0:
        print("  No tag data, skipping.")
        return result_df, []

    tags['trackable_name'] = tags['trackable_name'].str.lower().str.strip()
    top_tags = tags['trackable_name'].value_counts().head(15).index.tolist()

    tag_cols = []
    for tn in top_tags:
        col = f"tag_{tn.replace(' ', '_').replace('-', '_').replace('/', '_')}"
        result_df = add_binary_feature(result_df, tags, tn, col)
        tag_cols.append(col)

    print(f"  Tag features: {len(tag_cols)}")
    return result_df, tag_cols


def add_demographics(df_filtered, result_df):
    """Add age, sex, and country features."""
    demo = df_filtered[['user_id', 'age', 'sex', 'country']].drop_duplicates(subset='user_id')
    demo['age'] = pd.to_numeric(demo['age'], errors='coerce')
    demo['age'] = demo['age'].fillna(demo['age'].median())
    demo['sex_female'] = demo['sex'].map({'female': 1, 'male': 0}).fillna(0.5)
    demo['country_us'] = (demo['country'] == 'US').astype(int)
    demo = demo[['user_id', 'age', 'sex_female', 'country_us']]

    result_df = result_df.merge(demo, on='user_id', how='left')
    result_df['age'] = result_df['age'].fillna(result_df['age'].median())
    result_df['sex_female'] = result_df['sex_female'].fillna(0.5)
    result_df['country_us'] = result_df['country_us'].fillna(0)
    print(f"  Demographic features: age, sex_female, country_us")
    return result_df


def run_pipeline():
    """Execute the full preprocessing pipeline and save results."""
    print("=" * 60)
    print("FLAREDOWN DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_raw_data()

    print("\nStep 2: Filtering to user-days with food AND symptoms...")
    df_filtered, valid_days = filter_food_and_symptom_days(df)
    del df; gc.collect()

    print("\nStep 3: Creating flare target variable...")
    target_df = create_target(df_filtered, valid_days)

    print("\nStep 4: Pivoting food features...")
    result, food_cols = pivot_foods(df_filtered, target_df)

    print("\nStep 5: Pivoting treatment features...")
    result, treat_cols = pivot_treatments(df_filtered, result)

    print("\nStep 6: Pivoting weather features...")
    result, weather_cols = pivot_weather(df_filtered, result)

    print("\nStep 7: Pivoting tag features...")
    result, tag_cols = pivot_tags(df_filtered, result)

    print("\nStep 8: Adding demographics...")
    result = add_demographics(df_filtered, result)

    del df_filtered; gc.collect()

    # Collect feature columns
    cat_cols = [f"foodcat_{c}" for c in FOOD_CATEGORIES]
    feature_cols = food_cols + cat_cols + treat_cols + weather_cols + tag_cols + ['age', 'sex_female', 'country_us']
    feature_cols = [c for c in feature_cols if c in result.columns]

    print(f"\n{'=' * 60}")
    print(f"FINAL DATASET:")
    print(f"  Rows: {len(result):,}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Target: flare (0/1)")
    print(f"  Flare rate: {result['flare'].mean():.1%}")
    print(f"{'=' * 60}")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_cols = ['user_id', 'checkin_date', 'max_symptom_severity', 'flare'] + feature_cols
    save_cols = [c for c in save_cols if c in result.columns]
    result[save_cols].to_parquet(DATA_DIR / "processed.parquet", index=False)
    print(f"\nSaved to {DATA_DIR / 'processed.parquet'}")

    # 70/30 stratified split
    X = result[feature_cols].copy()
    y = result['flare'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    X_train.to_parquet(DATA_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(DATA_DIR / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(DATA_DIR / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(DATA_DIR / "y_test.parquet", index=False)
    pd.Series(feature_cols).to_csv(DATA_DIR / "feature_cols.csv", index=False, header=False)

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Features list: {DATA_DIR / 'feature_cols.csv'}")
    print("DONE!")

    return result, feature_cols


if __name__ == '__main__':
    run_pipeline()
