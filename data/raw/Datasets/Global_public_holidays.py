import pandas as pd
import holidays

# 1. Load your dataset
df = pd.read_csv('master_market_dataset_with_macro.csv')
df['Week_Start'] = pd.to_datetime(df['Week_Start'])
df['Year'] = df['Week_Start'].dt.year

# 2. Optimized function with caching to avoid re-generating holiday calendars every row
holiday_cache = {}


def get_holidays_for_week(row):
  country_code = row['Country_Code']
  week_start = row['Week_Start']
  year = row['Year']

  # Create a cache key for country and year
  cache_key = (country_code, year)
  if cache_key not in holiday_cache:
    try:
      # holidays.country_holidays natively accepts ISO2 codes like 'US', 'IN', etc.
      holiday_cache[cache_key] = holidays.country_holidays(
          country_code, years=year
      )
    except Exception:
      holiday_cache[cache_key] = {}

  country_holidays = holiday_cache[cache_key]
  week_range = pd.date_range(week_start, periods=7, freq='D')

  try:
    holiday_names = [
        country_holidays.get(d) for d in week_range if d in country_holidays
    ]
    holiday_names = [h for h in holiday_names if h]  # remove None values
    return {
        'is_holiday': int(len(holiday_names) > 0),
        'holiday_name': ', '.join(set(holiday_names))
        if holiday_names
        else None,  # Use set to avoid duplicate holiday names in a week
        'holiday_count': len(holiday_names),
    }
  except Exception:
    return {'is_holiday': 0, 'holiday_name': None, 'holiday_count': 0}


# 3. Apply the function across rows
print('⏳ Processing holidays for your dataset...')
holiday_info = df.apply(get_holidays_for_week, axis=1, result_type='expand')
df = pd.concat([df, holiday_info], axis=1)

# 4. Save the enriched dataset
output_file = 'master_market_dataset_with_holidays.csv'
df.to_csv(output_file, index=False)
print(f"✅ Successfully saved holiday features to '{output_file}'!")
print(
    df[[
        'Week_Start',
        'Country_Code',
        'is_holiday',
        'holiday_name',
        'holiday_count',
    ]].head(10)
)