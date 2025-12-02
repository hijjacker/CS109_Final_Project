import numpy as np
import pandas as pd

# =========================
# 1. Feature indices & names
# =========================

SURF_HOAR            = 0
DEPTH_HOAR           = 1
SLOPE_GT30           = 2
SLOPE_GT40           = 3
CONCAVE              = 4
ANCHORED             = 5
SUN_CRUST            = 6
WARMER               = 7
COLDER               = 8
AVYS_NEARBY          = 9
SNOW_30CM_24H        = 10
STRONG_WIND          = 11
WEAK_LAYER_SHALLOW   = 12
RAIN_ON_SNOW         = 13
RAPID_WARMING        = 14
SUN_EXPOSURE         = 15
ABOVE_TREELINE       = 16
EARLY_SEASON         = 17
OVERNIGHT_REFREEZE   = 18
DAYS_SINCE_STORM_GT3 = 19

feature_names = [
    "surface_hoar",
    "depth_hoar",
    "slope_gt_30",
    "slope_gt_40",
    "concave",
    "anchored",
    "sun_crust",
    "warmer",
    "colder",
    "avys_nearby",
    "snow_30cm_24h",
    "strong_wind",
    "weak_layer_shallow",
    "rain_on_snow",
    "rapid_warming",
    "sun_exposure",
    "above_treeline",
    "early_season",
    "overnight_refreeze",
    "days_since_storm_gt3",
]

# =========================
# 2. Regimes + probabilities
# =========================

regime_names = ["STORM", "COLD_CLEAR", "WARM_WET", "DRY_PERSISTENT"]
regime_probs = np.array([0.15, 0.25, 0.20, 0.40])  # must sum to 1

P = np.array([
    # STORM
    [
        0.05,  # 0 Surface hoar (usually buried or destroyed)
        0.10,  # 1 Depth hoar
        0.60,  # 2 Slope > 30
        0.25,  # 3 Slope > 40
        0.30,  # 4 Concave slope
        0.40,  # 5 Anchored terrain
        0.10,  # 6 Sun crust present
        0.30,  # 7 Warmer air temp
        0.10,  # 8 Colder air temp
        0.25,  # 9 Other avalanches nearby
        0.80,  # 10 >30cm snowfall (big storm days)
        0.70,  # 11 Strong transport winds
        0.40,  # 12 Weak layer within 60cm
        0.10,  # 13 Rain on snow
        0.10,  # 14 Rapid warming
        0.30,  # 15 Sun exposure
        0.50,  # 16 Above treeline
        0.40,  # 17 Early season (storms often early)
        0.20,  # 18 Overnight refreeze
        0.10,  # 19 >3 days since last storm
    ],
    # COLD_CLEAR
    [
        0.30,  # 0 Surface hoar (loves cold clear)
        0.15,  # 1 Depth hoar
        0.60,  # 2 Slope > 30
        0.25,  # 3 Slope > 40
        0.30,  # 4 Concave slope
        0.40,  # 5 Anchored terrain
        0.10,  # 6 Sun crust
        0.10,  # 7 Warmer
        0.80,  # 8 Colder
        0.05,  # 9 Avalanches nearby
        0.00,  # 10 >30cm 24h (no storm)
        0.20,  # 11 Strong winds
        0.35,  # 12 Weak layer within 60cm
        0.00,  # 13 Rain on snow
        0.00,  # 14 Rapid warming
        0.70,  # 15 Sun exposure
        0.50,  # 16 Above treeline
        0.25,  # 17 Early season
        0.70,  # 18 Overnight refreeze
        0.70,  # 19 >3 days since last storm
    ],
    # WARM_WET
    [
        0.05,  # 0 Surface hoar
        0.05,  # 1 Depth hoar
        0.60,  # 2 Slope > 30
        0.25,  # 3 Slope > 40
        0.30,  # 4 Concave slope
        0.40,  # 5 Anchored terrain
        0.35,  # 6 Sun crust
        0.80,  # 7 Warmer
        0.05,  # 8 Colder
        0.15,  # 9 Avalanches nearby
        0.15,  # 10 >30cm 24h (can be rainy storm)
        0.40,  # 11 Strong winds
        0.30,  # 12 Weak layer within 60cm
        0.40,  # 13 Rain on snow
        0.60,  # 14 Rapid warming
        0.80,  # 15 Sun exposure
        0.50,  # 16 Above treeline
        0.15,  # 17 Early season
        0.20,  # 18 Overnight refreeze
        0.50,  # 19 >3 days since last storm
    ],
    # DRY_PERSISTENT
    [
        0.25,  # 0 Surface hoar
        0.25,  # 1 Depth hoar
        0.60,  # 2 Slope > 30
        0.25,  # 3 Slope > 40
        0.30,  # 4 Concave slope
        0.40,  # 5 Anchored terrain
        0.20,  # 6 Sun crust
        0.40,  # 7 Warmer
        0.20,  # 8 Colder
        0.15,  # 9 Avalanches nearby
        0.05,  # 10 >30cm 24h
        0.30,  # 11 Strong winds
        0.50,  # 12 Weak layer within 60cm
        0.05,  # 13 Rain on snow
        0.10,  # 14 Rapid warming
        0.60,  # 15 Sun exposure
        0.50,  # 16 Above treeline
        0.20,  # 17 Early season
        0.30,  # 18 Overnight refreeze
        0.80,  # 19 >3 days since last storm
    ],
])

# =========================
# 3. Data generator (features)
# =========================

def generate_synthetic_avalanche_data(N, regime_probs, P, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_regimes, num_features = P.shape
    assert num_features == 20

    # Sample regimes for each day
    regime_idx = np.random.choice(num_regimes, size=N, p=regime_probs)

    # Sample features given regime
    X = np.zeros((N, num_features), dtype=int)
    for i in range(N):
        r = regime_idx[i]
        probs = P[r]
        X[i, :] = (np.random.rand(num_features) < probs).astype(int)

    # Logical cleanup
    for i in range(N):
        # If big storm today, then it hasn't been >3 days since last storm
        if X[i, SNOW_30CM_24H] == 1:
            X[i, DAYS_SINCE_STORM_GT3] = 0

        # Warmer and colder shouldn't both be 1
        if X[i, WARMER] == 1 or X[i, RAPID_WARMING] == 1:
            X[i, COLDER] = 0
        
        if X[i, COLDER] == 1:
            X[i, WARMER] == 0 or X[i, RAPID_WARMING] == 0

        # If rain on snow, force warmer = 1 and colder = 0
        if X[i, RAIN_ON_SNOW] == 1:
            X[i, WARMER] = 1
            X[i, COLDER] = 0

        # If above treeline, assume anchored terrain present (your modeling choice)
        if X[i, ABOVE_TREELINE] == 1:
            X[i, ANCHORED] = 1
        if X[i, ANCHORED] == 0 and X[i, ABOVE_TREELINE] == 1:
            X[i, ANCHORED] = 0
        
        # If greater than 30, then also greater than 40
        if X[i, SLOPE_GT40] == 1:
            X[i, SLOPE_GT30] == 1

    return X, regime_idx

# =========================
# 4. Red-flag + requirement labeling
# =========================

def meets_requirements(row, requirements):
    for req in requirements:
        if row[req] != 1:
            return False
    return True

def red_flags_to_prob(num_flags: int) -> float:
    if num_flags == 0:
        return 0.20
    elif num_flags == 1:
        return 0.40
    elif num_flags == 2:
        return 0.55
    elif num_flags == 3:
        return 0.65
    elif num_flags == 4:
        return 0.75
    else:  # 5 or more red flags
        return 0.90 

def generate_labels_red_flags(X, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N, D = X.shape
    assert D == 20, "Expecting 20 features"

    red_flag_idxs = [
        DEPTH_HOAR,
        SLOPE_GT30,
        SNOW_30CM_24H,
        STRONG_WIND,
        WEAK_LAYER_SHALLOW,
        RAIN_ON_SNOW,
        RAPID_WARMING,
        AVYS_NEARBY,
        SUN_EXPOSURE,
        OVERNIGHT_REFREEZE,
        WARMER
    ]

    requirements = [
        SLOPE_GT30,
        ABOVE_TREELINE
    ]

    y = np.zeros(N, dtype=int)

    for i in range(N):
        row = X[i, :]
        if not meets_requirements(row, requirements):
            y[i] = 0
            continue
        num_flags = X[i, red_flag_idxs].sum()
        p = red_flags_to_prob(num_flags)
        y[i] = int(np.random.rand() < p)

    return y

# =========================
# 5. Generate data & save CSV
# =========================

if __name__ == "__main__":
    N = 500  # number of synthetic days

    # Split into train and test
    train_num = int(N * 0.9)
    test_num = N - train_num

    train, regime_idx = generate_synthetic_avalanche_data(
        N=train_num,
        regime_probs=regime_probs,
        P=P,
        seed=109,
    )
    test, regime_idx = generate_synthetic_avalanche_data(
        N=test_num,
        regime_probs=regime_probs,
        P=P,
        seed=109,
    )

    y_train = generate_labels_red_flags(train, seed=42)
    y_test = generate_labels_red_flags(test, seed=42)

    # Build both DataFrame
    train_df = pd.DataFrame(train, columns=feature_names)
    train_df["avalanche"] = y_train
    test_df = pd.DataFrame(test, columns=feature_names)
    test_df["avalanche"] = y_test

    # Save to CSV
    train_out_filename = "train_avalanche_data.csv"
    test_out_filename = "test_avalanche_data.csv"
    train_df.to_csv(train_out_filename, index=False)
    test_df.to_csv(test_out_filename, index=False)