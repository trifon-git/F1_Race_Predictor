import requests
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime
import json
import os
import hashlib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://api.jolpi.ca/ergast/f1"
CACHE_DIR = "cache"  # Directory to store cached data

class F1DataFetcher:
    def __init__(self):
        # Initialize requests session
        self.session = requests.Session()
        
        # Add cache compression
        self.cache_config = {
            'qualifying': {'hours': 24, 'compress': True},
            'race': {'hours': 24, 'compress': True},
            'drivers': {'hours': 12, 'compress': False},
            'standings': {'hours': 6, 'compress': False}
        }
        
    def _write_cache(self, cache_path: Path, data: Dict):
        """Write data to cache file with optional compression."""
        try:
            cache_type = next((k for k in self.cache_config.keys() if k in str(cache_path)), None)
            should_compress = self.cache_config.get(cache_type, {}).get('compress', False)
            
            if should_compress:
                import gzip
                with gzip.open(str(cache_path) + '.gz', 'wt') as f:
                    json.dump(data, f)
            else:
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            logger.error(f"Error writing to cache file: {str(e)}")
    
    def _get_cache_path(self, endpoint: str) -> Path:
        """Generate a cache file path for the given endpoint."""
        # Create a hash of the endpoint to use as filename
        filename = hashlib.md5(endpoint.encode()).hexdigest() + '.json'
        return Path(CACHE_DIR) / filename
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 1) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        # Check if cache is older than max_age_hours
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - cache_time
        return age.total_seconds() < (max_age_hours * 3600)
    
    def _read_cache(self, cache_path: Path) -> Dict:
        """Read data from cache file."""
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache file: {str(e)}")
            return None
    
    def _write_cache(self, cache_path: Path, data: Dict):
        """Write data to cache file."""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error writing to cache file: {str(e)}")
    
    def _make_request(self, endpoint: str) -> Dict:
        """Make a GET request to the Jolpica F1 API with caching."""
        cache_path = self._get_cache_path(endpoint)
        
        # Try to get data from cache first
        if self._is_cache_valid(cache_path):
            cached_data = self._read_cache(cache_path)
            if cached_data:
                logger.info(f"Using cached data for {endpoint}")
                return cached_data
        
        # If cache is invalid or missing, make the API request
        try:
            logger.info(f"Fetching fresh data for {endpoint}")
            response = self.session.get(f"{BASE_URL}/{endpoint}.json")
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self._write_cache(cache_path, data)
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from {endpoint}: {str(e)}")
            
            # If request fails but we have cached data, use it as fallback
            if cache_path.exists():
                logger.info(f"Using cached data as fallback for {endpoint}")
                cached_data = self._read_cache(cache_path)
                if cached_data:
                    return cached_data
            
            raise
    
    def get_current_drivers(self) -> List[Dict]:
        """Get current F1 drivers with their constructors."""
        # Get drivers
        drivers_data = self._make_request("current/drivers")['MRData']['DriverTable']['Drivers']
        
        # Get driver standings to get current constructors
        standings_data = self._make_request("current/driverStandings")
        standings = standings_data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
        
        # Create a mapping of driver ID to constructor
        constructor_map = {
            standing['Driver']['driverId']: standing['Constructors'][0]['name']
            for standing in standings
        }
        
        # Format driver data
        formatted_drivers = []
        for driver in drivers_data:
            driver_id = driver['driverId']
            formatted_drivers.append({
                'driver_id': driver_id,
                'firstname': driver['givenName'],
                'lastname': driver['familyName'],
                'constructor': constructor_map.get(driver_id, 'Unknown')
            })
        
        return formatted_drivers
    
    def get_qualifying_results(self) -> pd.DataFrame:
        """Get qualifying results for the current season."""
        data = self._make_request("current/qualifying")
        races = data['MRData']['RaceTable']['Races']
        qualifying_results = []
        
        for race in races:
            race_name = race['raceName']
            circuit = race['Circuit']['circuitName']
            for quali in race['QualifyingResults']:
                qualifying_results.append({
                    'race_name': race_name,
                    'circuit': circuit,
                    'driver_id': quali['Driver']['driverId'],
                    'constructor': quali['Constructor']['name'],
                    'position': int(quali.get('position', 0)),
                    'q1_time': quali.get('Q1', None),
                    'q2_time': quali.get('Q2', None),
                    'q3_time': quali.get('Q3', None)
                })
        
        return pd.DataFrame(qualifying_results)
    
    def get_sprint_results(self) -> pd.DataFrame:
        """Get sprint race results for the current season."""
        try:
            data = self._make_request("current/sprint")
            races = data['MRData']['RaceTable']['Races']
            sprint_results = []
            
            for race in races:
                race_name = race['raceName']
                circuit = race['Circuit']['circuitName']
                for result in race['SprintResults']:
                    sprint_results.append({
                        'race_name': race_name,
                        'circuit': circuit,
                        'driver_id': result['Driver']['driverId'],
                        'constructor': result['Constructor']['name'],
                        'grid': int(result['grid']),
                        'position': int(result['position']) if result['position'].isdigit() else None,
                        'points': float(result['points']),
                        'laps': int(result['laps']),
                        'status': result['status']
                    })
            
            return pd.DataFrame(sprint_results)
        except:
            # Return empty DataFrame if no sprint races
            return pd.DataFrame()
    
    def get_current_season_results(self) -> pd.DataFrame:
        """Get current season race results."""
        # Get race data from API
        data = self._make_request("current/results")
        races = data['MRData']['RaceTable']['Races']
        results = []
        
        for race in races:
            race_name = race['raceName']
            circuit = race['Circuit']['circuitName']
            race_date = race['date']
            
            for result in race['Results']:
                fastest_lap = result.get('FastestLap', {})
                results.append({
                    'race_name': race_name,
                    'circuit': circuit,
                    'date': race_date,
                    'driver_id': result['Driver']['driverId'],
                    'constructor': result['Constructor']['name'],
                    'grid': int(result['grid']),
                    'position': int(result['position']) if result['position'].isdigit() else None,
                    'points': float(result['points']),
                    'laps': int(result['laps']),
                    'status': result['status'],
                    'fastest_lap_rank': fastest_lap.get('rank', None),
                    'fastest_lap_time': fastest_lap.get('Time', {}).get('time', None),
                    'fastest_lap_speed': fastest_lap.get('AverageSpeed', {}).get('speed', None)
                })
        
        return pd.DataFrame(results)
    
    def get_driver_standings(self) -> pd.DataFrame:
        """Get current driver standings."""
        data = self._make_request("current/driverStandings")
        standings = data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
        return pd.DataFrame(standings)
    
    def get_constructor_standings(self) -> pd.DataFrame:
        """Get current constructor standings."""
        data = self._make_request("current/constructorStandings")
        standings = data['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
        return pd.DataFrame(standings)
    
    def get_next_race(self) -> Dict:
        """Get information about the next race in the calendar."""
        try:
            # Get current season schedule
            data = self._make_request("current")
            races = data['MRData']['RaceTable']['Races']
            
            # Convert current date to required format
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Find the next race
            for race in races:
                race_date = race['date']
                if race_date > current_date:
                    return {
                        'name': race['raceName'],
                        'circuit': race['Circuit']['circuitName'],
                        'date': race_date,
                        'round': race['round']
                    }
            
            return None  # No upcoming races found
            
        except Exception as e:
            logger.error(f"Error fetching next race: {str(e)}")
            raise
    
    def get_circuits(self) -> List[str]:
        """Get all circuits in the current season."""
        try:
            # Get current season schedule
            data = self._make_request("current")
            races = data['MRData']['RaceTable']['Races']
            
            # Extract unique circuit names
            circuits = [race['Circuit']['circuitName'] for race in races]
            
            return sorted(circuits)  # Return sorted list for consistent display
            
        except Exception as e:
            logger.error(f"Error fetching circuits: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            cache_dir = Path(CACHE_DIR)
            if cache_dir.exists():
                for cache_file in cache_dir.glob('*.json'):
                    cache_file.unlink()
                logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    def _get_season_year(self, date_str: str) -> int:
        """Extract year from date string or Timestamp."""
        if isinstance(date_str, pd.Timestamp):
            return date_str.year
        elif isinstance(date_str, str):
            return int(date_str.split('-')[0])
        else:
            raise ValueError(f"Unexpected date format: {type(date_str)}")

    def get_last_n_races(self, n: int = 20) -> pd.DataFrame:
        """Get results from the last N races across seasons."""
        # Add batch processing
        BATCH_SIZE = 5
        current_season = datetime.now().year
        results = []
        
        for season in range(current_season, current_season - 3, -1):
            if len(results) >= n:
                break
                
            season_data = self._make_request(f"{season}/results")
            season_races = season_data['MRData']['RaceTable']['Races']
            
            # Process races in batches
            for i in range(0, len(season_races), BATCH_SIZE):
                batch = season_races[i:i + BATCH_SIZE]
                results.extend(self._process_race_results(batch))
                
                if len(results) >= n:
                    break
        
        # Convert to DataFrame and sort by date
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df['date'] = pd.to_datetime(results_df['date'])
            results_df = results_df.sort_values('date', ascending=False)
            results_df = results_df.head(n)  # Keep only the last n races
        
        return results_df
    
    def _process_race_results(self, races: List[Dict]) -> List[Dict]:
        """Helper method to process race results consistently."""
        results = []
        for race in races:
            race_name = race['raceName']
            circuit = race['Circuit']['circuitName']
            race_date = race['date']
            
            for result in race['Results']:
                fastest_lap = result.get('FastestLap', {})
                results.append({
                    'race_name': race_name,
                    'circuit': circuit,
                    'date': race_date,
                    'driver_id': result['Driver']['driverId'],
                    'constructor': result['Constructor']['name'],
                    'grid': int(result['grid']),
                    'position': int(result['position']) if result['position'].isdigit() else None,
                    'points': float(result['points']),
                    'laps': int(result['laps']),
                    'status': result['status'],
                    'fastest_lap_rank': fastest_lap.get('rank', None),
                    'fastest_lap_time': fastest_lap.get('Time', {}).get('time', None),
                    'fastest_lap_speed': fastest_lap.get('AverageSpeed', {}).get('speed', None)
                })
        return results
    
    def _calculate_team_trend(self, constructor_results: pd.DataFrame) -> float:
        """Calculate team's performance trend over the season."""
        if len(constructor_results) < 2:
            return 0.0
            
        positions = pd.to_numeric(constructor_results['position'], errors='coerce')
        races = range(len(positions))
        
        # Calculate trend using linear regression
        z = np.polyfit(races, positions, 1)
        return -z[0]  # Negative slope means improving trend

    def get_qualifying_results_for_races(self, race_dates: List[str]) -> pd.DataFrame:
        """Get qualifying results for specific races by their dates."""
        qualifying_results = []
        
        # Group race dates by season
        seasons = {}
        for date in race_dates:
            year = self._get_season_year(date)
            if year not in seasons:
                seasons[year] = []
            seasons[year].append(date)
        
        # Fetch qualifying results for each season
        for year, dates in seasons.items():
            endpoint = "current/qualifying" if year == datetime.now().year else f"{year}/qualifying"
            try:
                data = self._make_request(endpoint)
                races = data['MRData']['RaceTable']['Races']
                
                for race in races:
                    if race['date'] in dates:
                        race_name = race['raceName']
                        circuit = race['Circuit']['circuitName']
                        for quali in race['QualifyingResults']:
                            qualifying_results.append({
                                'race_name': race_name,
                                'circuit': circuit,
                                'date': race['date'],
                                'driver_id': quali['Driver']['driverId'],
                                'constructor': quali['Constructor']['name'],
                                'position': int(quali.get('position', 0)),
                                'q1_time': quali.get('Q1', None),
                                'q2_time': quali.get('Q2', None),
                                'q3_time': quali.get('Q3', None)
                            })
            except Exception as e:
                logger.error(f"Error fetching qualifying data for {year}: {str(e)}")
        
        return pd.DataFrame(qualifying_results)

    def get_sprint_results_for_races(self, race_dates: List[str]) -> pd.DataFrame:
        """Get sprint results for specific races by their dates."""
        sprint_results = []
        
        # Group race dates by season
        seasons = {}
        for date in race_dates:
            year = self._get_season_year(date)
            if year not in seasons:
                seasons[year] = []
            seasons[year].append(date)
        
        # Fetch sprint results for each season
        for year, dates in seasons.items():
            endpoint = "current/sprint" if year == datetime.now().year else f"{year}/sprint"
            try:
                data = self._make_request(endpoint)
                races = data['MRData']['RaceTable']['Races']
                
                for race in races:
                    if race['date'] in dates:
                        race_name = race['raceName']
                        circuit = race['Circuit']['circuitName']
                        for result in race['SprintResults']:
                            sprint_results.append({
                                'race_name': race_name,
                                'circuit': circuit,
                                'date': race['date'],
                                'driver_id': result['Driver']['driverId'],
                                'constructor': result['Constructor']['name'],
                                'grid': int(result['grid']),
                                'position': int(result['position']) if result['position'].isdigit() else None,
                                'points': float(result['points']),
                                'laps': int(result['laps']),
                                'status': result['status']
                            })
            except:
                # Skip if no sprint races for this season/race
                continue
        
        return pd.DataFrame(sprint_results)

    def _calculate_team_trend(self, constructor_results: pd.DataFrame) -> float:
        """Calculate team's performance trend over the season."""
        if len(constructor_results) < 2:
            return 0.0
            
        positions = pd.to_numeric(constructor_results['position'], errors='coerce')
        races = range(len(positions))
        
        # Calculate trend using linear regression
        z = np.polyfit(races, positions, 1)
        return -z[0]  # Negative slope means improving trend

if __name__ == "__main__":
    # Test the data fetcher
    fetcher = F1DataFetcher()
    try:
        # First request will fetch from API and cache
        logger.info("First request (will fetch from API):")
        current_drivers = fetcher.get_current_drivers()
        season_results = fetcher.get_current_season_results()
        qualifying_results = fetcher.get_qualifying_results()
        sprint_results = fetcher.get_sprint_results()
        
        # Second request should use cached data
        logger.info("\nSecond request (should use cache):")
        current_drivers = fetcher.get_current_drivers()
        season_results = fetcher.get_current_season_results()
        qualifying_results = fetcher.get_qualifying_results()
        sprint_results = fetcher.get_sprint_results()
        
        print("\nData fetched successfully!")
        print(f"Number of current drivers: {len(current_drivers)}")
        print(f"Number of race results: {len(season_results)}")
        print(f"Number of qualifying results: {len(qualifying_results)}")
        print(f"Number of sprint results: {len(sprint_results)}")
        
        # Test cache clearing
        fetcher.clear_cache()
        print("\nCache cleared successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}")