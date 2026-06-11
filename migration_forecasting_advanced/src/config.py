APP_TITLE = "International Migration Forecasting Intelligence"
DEFAULT_DATE_COLUMN = "date"
DEFAULT_GROUP_COLUMN = "country"
DEFAULT_TARGET_COLUMN = "migration_count"
SAMPLE_COUNTRIES = ["Canada", "Germany", "United Kingdom", "Australia", "United States"]
SUPPORTED_MODELS = [
    "Naive",
    "Seasonal Naive",
    "Moving Average",
    "Exponential Smoothing",
    "SARIMAX",
    "Random Forest",
    "Prophet (optional)",
    "LSTM (optional)",
]
