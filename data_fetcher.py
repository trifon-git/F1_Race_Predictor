import fastf1
import pandas as pd
import os

# Enable FastF1 cache
FASTF1_CACHE_DIR = './fastf1_cache'
os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)

def get_last_n_races_results(n=5):
    current_year = pd.Timestamp.now().year
    all_results = []

    for year in range(current_year, current_year-3, -1):
        try:
            schedule = fastf1.get_event_schedule(year)
            schedule['EventDate'] = pd.to_datetime(schedule['EventDate'], errors='coerce', utc=True)
            schedule = schedule.dropna(subset=['EventDate'])
            schedule = schedule[schedule['EventDate'] < pd.Timestamp.now(tz='UTC')]
            schedule = schedule.sort_values('EventDate', ascending=False)

            for _, race in schedule.iterrows():
                if len(all_results) >= n:
                    break

                try:
                    session = fastf1.get_session(year, race['RoundNumber'], 'R')
                    session.load()
                    race_results = session.results

                    df = pd.DataFrame({
                        'DriverId': race_results['Abbreviation'],
                        'Position': race_results['Position'],
                        'GridPosition': race_results['GridPosition'],
                        'Status': race_results['Status'],
                        'EventDate': race['EventDate'],
                        'EventName': race['EventName'],
                        'CircuitName': race['Location'],
                        'Year': year,
                        'RoundNumber': race['RoundNumber']
                    })

                    all_results.append(df)

                except Exception as e:
                    print(f"Failed to fetch race {race['EventName']}: {e}")

            if len(all_results) >= n:
                break

        except Exception as e:
            print(f"Error processing year {year}: {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)

    return pd.DataFrame()

def get_next_race_info(year: int = pd.Timestamp.now().year) -> pd.Series | None:
    """Get information about the next upcoming race."""
    try:
        schedule = fastf1.get_event_schedule(year)
        if schedule.empty:
            return None
        next_race = schedule[schedule['EventDate'].dt.tz_localize(None) > pd.Timestamp.now()].iloc[0]
        return next_race
    except Exception as e:
        print(f"Error getting next race: {e}")
        return None
