import pandas as pd


def detect_anomalies(values):
    """
    Detect unusual readings using the Z-score method.
    Returns True for anomalies and False for normal readings.
    """
    mean = values.mean()
    std = values.std()

    if pd.isna(std) or std == 0:
        return pd.Series(False, index=values.index)

    z_scores = ((values - mean) / std).abs()

    return z_scores > 2


def detect_rolling_anomalies(values, window=5, threshold=3.5):
    """
    Compare each reading with the preceding readings.
    Returns True when a reading is unusually far from
    the recent historical median.
    """
    history = values.shift(1)

    rolling_median = history.rolling(
        window=window,
        min_periods=3
    ).median()

    rolling_mad = history.rolling(
        window=window,
        min_periods=3
    ).apply(
        lambda x: pd.Series(x).sub(pd.Series(x).median()).abs().median(),
        raw=True
    )

    # Convert median absolute deviation into a robust scale estimate
    robust_scale = 1.4826 * rolling_mad

    scores = (values - rolling_median).abs() / robust_scale

    # If recent readings are identical, handle sudden changes separately
    zero_scale_anomaly = (
        robust_scale.eq(0)
        & rolling_median.notna()
        & values.ne(rolling_median)
    )

    return (
        (scores > threshold) | zero_scale_anomaly
    ).fillna(False)




def explain_anomalies(values):
    mean = values.mean()
    std = values.std()

    if pd.isna(std) or std == 0:
        return pd.Series(
            ["Readings are consistent with the available data."] * len(values),
            index=values.index
        )

    z_scores = ((values - mean) / std).abs()

    explanations = []

    for value, score in zip(values, z_scores):
        if score > 3:
            explanations.append(
                f"High alert: {value} is far from the usual average "
                f"of {mean:.2f}. This may indicate an unusual event."
            )
        elif score > 2:
            explanations.append(
                f"Warning: {value} differs noticeably from the usual "
                f"average of {mean:.2f}. Check for possible unusual activity."
            )
        else:
            explanations.append(
                f"{value} is within the expected range based on the "
                f"overall data."
            )

    return pd.Series(explanations, index=values.index)


if __name__ == "__main__":
    readings = pd.Series([
        100, 102, 98, 101, 99, 103, 100, 102, 500, 101
    ])

    results = detect_anomalies(readings)
    rolling_results = detect_rolling_anomalies(readings)

    print("\nRolling detector results:")
    print("Anomaly flags:", rolling_results.tolist())
    print("Detected anomalies:", int(rolling_results.sum()))
    explanations = explain_anomalies(readings)

    print("Readings:", readings.tolist())
    print("Anomaly flags:", results.tolist())
    print("Detected anomalies:", int(results.sum()))

    print("\nAnomaly explanations:")
    for reading, is_anomaly, explanation in zip(
        readings, results, explanations
    ):
        if is_anomaly:
            print(f"Reading {reading}: {explanation}")

    print("Readings:", readings.tolist())
    print("Anomaly flags:", results.tolist())
    print("Detected anomalies:", int(results.sum()))