import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from math import comb  # Import comb from math module
import itertools
import warnings

warnings.filterwarnings('ignore')


class F1FantasyPredictor:
    def __init__(self, data_path='./', budget=100):
        self.data_path = data_path
        self.budget = budget

        # Store pricing for constructors and drivers
        self.constructor_pricing = {
            "Mclaren": 32,
            "Mercedes": 24,
            "Ferrari": 27,
            "Redbull": 23,
            "Williams": 15,
            "Racing Bulls": 20,
            "Haas": 13,
            "Alpine": 12,
            "Aston Martin": 14,
            "Stake": 16
        }

        self.driver_pricing = {
            "Oscar": 32,
            "Lando": 27,
            "Max": 25,
            "George": 22,
            "Kimi": 20,
            "Charles": 23,
            "Albon": 15,
            "Yuki": 8.5,
            "Isack": 14,
            "Lewis": 21,
            "Pierre": 6.5,
            "Carlos": 9,
            "Bearman": 7,
            "Ocon": 12,
            "Alonso": 16,
            "Stroll": 5.5,  # Update for new driver and their prices (Updated for Britan)
            "Lawson": 10,
            "Hulk": 13,
            "Bortoleto": 11,
            "Colapinto": 6
        }

        # Map the drivers to their full names and teams (2025 season assumption)
        self.driver_mapping = {
            "Oscar": {"full_name": "Oscar Piastri", "team": "Mclaren"},
            "Lando": {"full_name": "Lando Norris", "team": "Mclaren"},
            "Max": {"full_name": "Max Verstappen", "team": "Redbull"},
            "George": {"full_name": "George Russell", "team": "Mercedes"},
            "Kimi": {"full_name": "Kimi Antonelli", "team": "Mercedes"},
            "Charles": {"full_name": "Charles Leclerc", "team": "Ferrari"},
            "Albon": {"full_name": "Alexander Albon", "team": "Williams"},
            "Yuki": {"full_name": "Yuki Tsunoda", "team": "Racing Bulls"},
            "Isack": {"full_name": "Isack Hadjar", "team": "Racing Bulls"},
            "Lewis": {"full_name": "Lewis Hamilton", "team": "Ferrari"},
            "Pierre": {"full_name": "Pierre Gasly", "team": "Alpine"},
            "Carlos": {"full_name": "Carlos Sainz", "team": "Williams"},
            "Bearman": {"full_name": "Oliver Bearman", "team": "Haas"},
            "Ocon": {"full_name": "Esteban Ocon", "team": "Haas"},
            "Alonso": {"full_name": "Fernando Alonso", "team": "Aston Martin"},
            "Stroll": {"full_name": "Lance Stroll", "team": "Aston Martin"},
            "Lawson": {"full_name": "Liam Lawson", "team": "Redbull"},
            "Hulk": {"full_name": "Nico Hülkenberg", "team": "Stake"},
            "Bortoleto": {"full_name": "Gabriel Bortoleto", "team": "Stake"},
            "Colapinto": {"full_name": "Franco Colapinto", "team": "Alpine"}
        }

        # Initialize empty DataFrames for loaded data
        self.races = None
        self.drivers = None
        self.constructors = None
        self.results = None
        self.qualifying = None
        self.circuits = None
        self.constructor_results = None
        self.constructor_standings = None
        self.driver_standings = None
        self.lap_times = None
        self.pit_stops = None

        # Store prediction models
        self.driver_model = None
        self.constructor_model = None

        # Store processed features for prediction
        self.driver_features = None
        self.constructor_features = None

    def load_data(self):
        """Load all necessary CSV data files"""
        try:
            self.races = pd.read_csv(f'{self.data_path}races.csv')
            self.drivers = pd.read_csv(f'{self.data_path}drivers.csv')
            self.constructors = pd.read_csv(f'{self.data_path}constructors.csv')
            self.results = pd.read_csv(f'{self.data_path}results.csv')
            self.qualifying = pd.read_csv(f'{self.data_path}qualifying.csv')
            self.circuits = pd.read_csv(f'{self.data_path}circuits.csv')
            self.constructor_results = pd.read_csv(f'{self.data_path}constructor_results.csv')
            self.constructor_standings = pd.read_csv(f'{self.data_path}constructor_standings.csv')
            self.driver_standings = pd.read_csv(f'{self.data_path}driver_standings.csv')
            self.lap_times = pd.read_csv(f'{self.data_path}lap_times.csv')
            self.pit_stops = pd.read_csv(f'{self.data_path}pit_stops.csv')

            print("Successfully loaded all data files")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def map_current_names(self):
        """
        Map historical driver and constructor data to current names
        This is necessary as the dataset might use different names than our pricing data
        """
        # Create reverse mappings for lookup
        self.driver_name_to_id = {}
        for driver_id, row in self.drivers.iterrows():
            full_name = f"{row['forename']} {row['surname']}"
            self.driver_name_to_id[full_name.lower()] = row['driverId']

        # Map current drivers/constructors to historical IDs
        self.current_driver_ids = {}
        for short_name, info in self.driver_mapping.items():
            full_name = info['full_name'].lower()
            if full_name in self.driver_name_to_id:
                self.current_driver_ids[short_name] = self.driver_name_to_id[full_name]
            else:
                # Handle new drivers not in historical data
                print(f"Warning: No historical data for {info['full_name']}")
                self.current_driver_ids[short_name] = None

        # Create constructor mapping
        self.current_constructor_ids = {}
        for constructor_ref, row in self.constructors.iterrows():
            for team_name in self.constructor_pricing.keys():
                # Check if constructor name matches (case insensitive partial match)
                if team_name.lower() in row['name'].lower():
                    self.current_constructor_ids[team_name] = row['constructorId']
                    break

    def find_circuit_by_name(self, circuit_name="Imola"):
        """Find circuit ID and information by name"""
        for _, circuit in self.circuits.iterrows():
            if circuit_name.lower() in circuit['name'].lower() or circuit_name.lower() in circuit['location'].lower():
                return circuit
        return None

    def get_circuit_historical_performance(self, circuit_id):
        """Get historical performance at a specific circuit"""
        # Find all races at this circuit
        circuit_races = self.races[self.races['circuitId'] == circuit_id]

        # Get all race IDs for this circuit
        race_ids = circuit_races['raceId'].tolist()

        # Get all results from these races
        circuit_results = self.results[self.results['raceId'].isin(race_ids)]

        # Join with driver and constructor data
        circuit_results = circuit_results.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId')
        circuit_results = circuit_results.merge(self.constructors[['constructorId', 'name']], on='constructorId')

        # Add race year
        circuit_results = circuit_results.merge(self.races[['raceId', 'year']], on='raceId')

        # Sort by year (descending) and position (ascending)
        circuit_results = circuit_results.sort_values(['year', 'positionOrder'], ascending=[False, True])

        return circuit_results

    def analyze_recent_form(self, n_races=5):
        """Analyze recent form for all drivers and constructors"""
        # Get the most recent n races
        recent_races = self.races.sort_values('date', ascending=False).head(n_races)
        race_ids = recent_races['raceId'].tolist()

        # Get results from these races
        recent_results = self.results[self.results['raceId'].isin(race_ids)]

        # Calculate average points and positions for each driver and constructor
        driver_form = recent_results.groupby('driverId').agg({
            'points': 'mean',
            'positionOrder': 'mean',
            'grid': 'mean'
        }).reset_index()

        constructor_form = recent_results.groupby('constructorId').agg({
            'points': 'mean',
            'positionOrder': 'mean',
            'grid': 'mean'
        }).reset_index()

        # Join with names
        driver_form = driver_form.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId')
        driver_form['full_name'] = driver_form['forename'] + ' ' + driver_form['surname']

        constructor_form = constructor_form.merge(self.constructors[['constructorId', 'name']], on='constructorId')

        return driver_form, constructor_form

    def prepare_feature_engineering(self, track_name="Imola"):
        """Engineer features for prediction model"""
        # Get circuit performance at specified track
        selected_circuit = self.find_circuit_by_name(track_name)
        if selected_circuit is None:
            print(f"{track_name} circuit not found, using generic model")
            track_performance = pd.DataFrame()
        else:
            print(f"Found {track_name} circuit, analyzing historical performance")
            track_performance = self.get_circuit_historical_performance(selected_circuit['circuitId'])

        # Get recent form
        driver_form, constructor_form = self.analyze_recent_form(n_races=10)

        # Create car performance metrics
        car_performance = {}
        for team_name, _ in self.constructor_pricing.items():
            # Find constructor ID
            constructor_id = None
            for _, constructor in self.constructors.iterrows():
                if team_name.lower() in constructor['name'].lower():
                    constructor_id = constructor['constructorId']
                    break

            # Get car performance metrics from constructor data
            if constructor_id is not None:
                # Recent performance (last 5 races)
                recent_constructor_results = self.constructor_results[
                    self.constructor_results['constructorId'] == constructor_id].sort_values('constructorResultsId',
                                                                                             ascending=False).head(5)
                avg_constructor_points = recent_constructor_results[
                    'points'].mean() if not recent_constructor_results.empty else 0

                # Track-specific performance
                if not track_performance.empty:
                    track_team_stats = track_performance[track_performance['constructorId'] == constructor_id]
                    track_avg_points = track_team_stats['points'].mean() if not track_team_stats.empty else 0
                else:
                    track_avg_points = 0

                # Track type analysis - some cars perform better on specific track types
                track_type_factor = 1.0  # Default
                if selected_circuit is not None:
                    # Analyze if track is high downforce, street, or high speed
                    track_location = selected_circuit['location'].lower() if 'location' in selected_circuit else ''
                    track_name_lower = selected_circuit['name'].lower() if 'name' in selected_circuit else ''

                    # High downforce tracks
                    if any(x in track_name_lower or x in track_location for x in ['monaco', 'hungary', 'singapore']):
                        # Teams known for good downforce (adjust based on 2024-2025 performance)
                        if team_name in ['Mclaren', 'Ferrari']:
                            track_type_factor = 1.15

                    # High speed tracks
                    if any(x in track_name_lower or x in track_location for x in ['monza', 'spa', 'baku']):
                        # Teams known for straight line speed
                        if team_name in ['Redbull', 'Mercedes']:
                            track_type_factor = 1.12

                # Combine factors for overall car performance
                car_performance[team_name] = {
                    'recent_points': avg_constructor_points,
                    'track_points': track_avg_points,
                    'track_type_factor': track_type_factor,
                    # Overall car score - weighted combination
                    'car_score': (avg_constructor_points * 0.6 + track_avg_points * 0.4) * track_type_factor
                }
            else:
                # Default values if no historical data
                car_performance[team_name] = {
                    'recent_points': 0,
                    'track_points': 0,
                    'track_type_factor': 1.0,
                    'car_score': 0
                }

        # Combine features for driver model
        driver_features = []
        for short_name, driver_info in self.driver_mapping.items():
            full_name = driver_info['full_name']
            team = driver_info['team']
            price = self.driver_pricing[short_name]

            # Find driver ID from full name
            driver_id = None
            for _, driver in self.drivers.iterrows():
                if f"{driver['forename']} {driver['surname']}".lower() == full_name.lower():
                    driver_id = driver['driverId']
                    break

            # Get recent form
            recent_stats = driver_form[driver_form['driverId'] == driver_id] if driver_id in driver_form[
                'driverId'].values else None

            # Get track performance
            track_stats = track_performance[
                track_performance['driverId'] == driver_id] if driver_id and not track_performance.empty else None

            # Calculate value metrics
            recent_avg_points = recent_stats['points'].values[
                0] if recent_stats is not None and not recent_stats.empty else 0
            recent_avg_position = recent_stats['positionOrder'].values[
                0] if recent_stats is not None and not recent_stats.empty else 20
            recent_avg_grid = recent_stats['grid'].values[
                0] if recent_stats is not None and not recent_stats.empty else 20

            track_avg_points = track_stats['points'].mean() if track_stats is not None and not track_stats.empty else 0

            # Get standings position
            if driver_id:
                latest_standing = self.driver_standings[self.driver_standings['driverId'] == driver_id].sort_values(
                    'raceId', ascending=False).head(1)
                standings_position = latest_standing['position'].values[0] if not latest_standing.empty else 20
            else:
                standings_position = 20

            # Get car performance factor
            car_factor = car_performance.get(team, {'car_score': 0})['car_score']

            # Calculate track-specific car advantage
            track_car_advantage = car_performance.get(team, {'track_type_factor': 1.0})['track_type_factor']

            # Calculate predicted points with car performance factor included
            # Driver skill (recent form + track history) + car performance
            driver_skill_factor = 0.65  # Weight for driver's personal performance
            car_skill_factor = 0.35  # Weight for car performance

            # Raw driver performance (without car factor)
            raw_driver_performance = (recent_avg_points * 0.7 + track_avg_points * 0.3)

            # Adjust prediction with car performance
            predicted_points = (raw_driver_performance * driver_skill_factor) + (car_factor * car_skill_factor)

            # Apply track-specific adjustment
            predicted_points = predicted_points * track_car_advantage

            # Calculate value (points per million)
            value = predicted_points / price if price > 0 else 0

            # Store features
            driver_features.append({
                'short_name': short_name,
                'full_name': full_name,
                'team': team,
                'price': price,
                'recent_avg_points': recent_avg_points,
                'recent_avg_position': recent_avg_position,
                'recent_avg_grid': recent_avg_grid,
                'track_avg_points': track_avg_points,
                'circuit_avg_points': track_avg_points,  # Add this to match training feature names
                'standings_position': standings_position,
                'car_performance': car_factor,
                'track_car_advantage': track_car_advantage,
                'predicted_points': predicted_points,
                'value': value
            })

        # Convert to DataFrame
        driver_features_df = pd.DataFrame(driver_features)

        # Similar process for constructors
        constructor_features = []
        for team_name, price in self.constructor_pricing.items():
            # Find constructor ID
            constructor_id = None
            for _, constructor in self.constructors.iterrows():
                if team_name.lower() in constructor['name'].lower():
                    constructor_id = constructor['constructorId']
                    break

            # Get recent form
            recent_stats = constructor_form[constructor_form['constructorId'] == constructor_id] if constructor_id in \
                                                                                                    constructor_form[
                                                                                                        'constructorId'].values else None

            # Get track performance
            track_team_stats = track_performance[track_performance[
                                                     'constructorId'] == constructor_id] if constructor_id and not track_performance.empty else None

            # Calculate value metrics
            recent_avg_points = recent_stats['points'].values[
                0] if recent_stats is not None and not recent_stats.empty else 0
            recent_avg_position = recent_stats['positionOrder'].values[
                0] if recent_stats is not None and not recent_stats.empty else 10

            track_avg_points = track_team_stats[
                'points'].mean() if track_team_stats is not None and not track_team_stats.empty else 0

            # Get standings position
            if constructor_id:
                latest_standing = self.constructor_standings[
                    self.constructor_standings['constructorId'] == constructor_id].sort_values('raceId',
                                                                                               ascending=False).head(1)
                standings_position = latest_standing['position'].values[0] if not latest_standing.empty else 10
            else:
                standings_position = 10

            # Get track-specific advantage
            track_factor = car_performance.get(team_name, {'track_type_factor': 1.0})['track_type_factor']

            # Calculate predicted points with track factor
            base_points = (
                                  recent_avg_points * 0.6 + track_avg_points * 0.4) * 2  # Constructors typically get double points
            predicted_points = base_points * track_factor

            # Calculate value (points per million)
            value = predicted_points / price if price > 0 else 0

            # Store features
            constructor_features.append({
                'team_name': team_name,
                'price': price,
                'recent_avg_points': recent_avg_points,
                'recent_avg_position': recent_avg_position,
                'track_avg_points': track_avg_points,
                'circuit_avg_points': track_avg_points,  # Add this to match training feature names
                'standings_position': standings_position,
                'track_factor': track_factor,
                'predicted_points': predicted_points,
                'value': value
            })

        # Convert to DataFrame
        constructor_features_df = pd.DataFrame(constructor_features)

        self.driver_features = driver_features_df
        self.constructor_features = constructor_features_df

        return driver_features_df, constructor_features_df

    def train_prediction_models(self):
        """Train ML models to predict driver and constructor points"""
        if self.driver_features is None or self.constructor_features is None:
            self.prepare_feature_engineering()

        print("Training prediction models using historical data...")

        # Create historical training dataset for drivers
        historical_driver_data = []

        # Get the most recent races (excluding the current season)
        recent_seasons = sorted(self.races['year'].unique())[-3:]  # Last 3 seasons
        season_races = self.races[self.races['year'].isin(recent_seasons)]

        # For each race, extract features and actual points
        for _, race in season_races.iterrows():
            race_id = race['raceId']
            circuit_id = race['circuitId']
            race_year = race['year']

            # Get results for this race
            race_results = self.results[self.results['raceId'] == race_id]

            # For each driver in the race
            for _, result in race_results.iterrows():
                driver_id = result['driverId']
                constructor_id = result['constructorId']
                actual_points = result['points']

                # Get driver's performance in previous races this season
                prev_races = self.races[(self.races['year'] == race_year) & (self.races['raceId'] < race_id)]
                prev_race_ids = prev_races['raceId'].tolist()

                if len(prev_race_ids) >= 3:  # Need at least 3 previous races for meaningful stats
                    # Get driver's previous results
                    prev_results = self.results[(self.results['driverId'] == driver_id) &
                                                (self.results['raceId'].isin(prev_race_ids))]

                    if not prev_results.empty:
                        # Calculate features
                        recent_avg_points = prev_results['points'].mean()
                        recent_avg_position = prev_results['positionOrder'].mean()

                        # Get qualifying performance
                        prev_quali = self.qualifying[(self.qualifying['driverId'] == driver_id) &
                                                     (self.qualifying['raceId'].isin(prev_race_ids))]
                        recent_avg_grid = prev_quali['position'].mean() if not prev_quali.empty else 15

                        # Get historical performance at this circuit
                        circuit_races = self.races[self.races['circuitId'] == circuit_id]
                        circuit_race_ids = circuit_races[circuit_races['raceId'] < race_id]['raceId'].tolist()

                        circuit_results = self.results[(self.results['driverId'] == driver_id) &
                                                       (self.results['raceId'].isin(circuit_race_ids))]

                        circuit_avg_points = circuit_results['points'].mean() if not circuit_results.empty else 0

                        # Get driver standings before this race
                        prev_standings = self.driver_standings[(self.driver_standings['driverId'] == driver_id) &
                                                               (self.driver_standings['raceId'].isin(prev_race_ids))]

                        if not prev_standings.empty:
                            latest_standing = prev_standings.sort_values('raceId', ascending=False).iloc[0]
                            standings_position = latest_standing['position']

                            # Find constructor and approximate price based on standings
                            constructor = \
                                self.constructors[self.constructors['constructorId'] == constructor_id].iloc[0]['name']

                            # Approximate price based on performance
                            driver_price = max(30 - standings_position, 5)  # Higher ranking -> higher price

                            # Store training example
                            historical_driver_data.append({
                                'price': driver_price,
                                'recent_avg_points': recent_avg_points,
                                'recent_avg_position': recent_avg_position,
                                'recent_avg_grid': recent_avg_grid,
                                'circuit_avg_points': circuit_avg_points,
                                'standings_position': standings_position,
                                'actual_points': actual_points
                            })

        # Create training dataset for constructors
        historical_constructor_data = []

        # For each race, extract features and actual points for constructors
        for _, race in season_races.iterrows():
            race_id = race['raceId']
            circuit_id = race['circuitId']
            race_year = race['year']

            # Get constructor results for this race
            constructor_race_results = self.constructor_results[self.constructor_results['raceId'] == race_id]

            # For each constructor in the race
            for _, result in constructor_race_results.iterrows():
                constructor_id = result['constructorId']
                actual_points = result['points']

                # Get constructor's performance in previous races this season
                prev_races = self.races[(self.races['year'] == race_year) & (self.races['raceId'] < race_id)]
                prev_race_ids = prev_races['raceId'].tolist()

                if len(prev_race_ids) >= 3:  # Need at least 3 previous races for meaningful stats
                    # Get constructor's previous results
                    prev_results = self.constructor_results[
                        (self.constructor_results['constructorId'] == constructor_id) &
                        (self.constructor_results['raceId'].isin(prev_race_ids))]

                    if not prev_results.empty:
                        # Calculate features
                        recent_avg_points = prev_results['points'].mean()

                        # Get driver results for this constructor to calculate avg position
                        team_drivers = self.results[(self.results['constructorId'] == constructor_id) &
                                                    (self.results['raceId'].isin(prev_race_ids))]

                        recent_avg_position = team_drivers['positionOrder'].mean() if not team_drivers.empty else 10

                        # Get historical performance at this circuit
                        circuit_races = self.races[self.races['circuitId'] == circuit_id]
                        circuit_race_ids = circuit_races[circuit_races['raceId'] < race_id]['raceId'].tolist()

                        circuit_results = self.constructor_results[
                            (self.constructor_results['constructorId'] == constructor_id) &
                            (self.constructor_results['raceId'].isin(circuit_race_ids))]

                        circuit_avg_points = circuit_results['points'].mean() if not circuit_results.empty else 0

                        # Get constructor standings before this race
                        prev_standings = self.constructor_standings[
                            (self.constructor_standings['constructorId'] == constructor_id) &
                            (self.constructor_standings['raceId'].isin(prev_race_ids))]

                        if not prev_standings.empty:
                            latest_standing = prev_standings.sort_values('raceId', ascending=False).iloc[0]
                            standings_position = latest_standing['position']

                            # Approximate price based on standings
                            constructor_price = max(30 - standings_position * 2, 10)  # Higher ranking -> higher price

                            # Store training example
                            historical_constructor_data.append({
                                'price': constructor_price,
                                'recent_avg_points': recent_avg_points,
                                'recent_avg_position': recent_avg_position,
                                'circuit_avg_points': circuit_avg_points,
                                'standings_position': standings_position,
                                'actual_points': actual_points
                            })

        # Convert to DataFrames
        driver_train_df = pd.DataFrame(historical_driver_data)
        constructor_train_df = pd.DataFrame(historical_constructor_data)

        print(
            f"Created training dataset with {len(driver_train_df)} driver samples and {len(constructor_train_df)} constructor samples")

        # Train driver model
        if not driver_train_df.empty:
            X_driver_train = driver_train_df[['price', 'recent_avg_points', 'recent_avg_position',
                                              'recent_avg_grid', 'circuit_avg_points', 'standings_position']]
            y_driver_train = driver_train_df['actual_points']

            # Split into train/test sets
            X_driver_train, X_driver_test, y_driver_train, y_driver_test = train_test_split(
                X_driver_train, y_driver_train, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_driver_train_scaled = scaler.fit_transform(X_driver_train)
            X_driver_test_scaled = scaler.transform(X_driver_test)

            # Train model with hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }

            grid_search = GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error'
            )

            grid_search.fit(X_driver_train_scaled, y_driver_train)

            # Get best model
            self.driver_model = grid_search.best_estimator_

            # Evaluate model
            y_pred = self.driver_model.predict(X_driver_test_scaled)
            mse = mean_squared_error(y_driver_test, y_pred)
            print(f"Driver model MSE: {mse:.2f}, RMSE: {np.sqrt(mse):.2f}")

            # Prepare for prediction on current data
            self.driver_scaler = scaler
        else:
            # Fallback to simpler model if no historical data
            X_driver = self.driver_features[['price', 'recent_avg_points', 'recent_avg_position',
                                             'recent_avg_grid', 'circuit_avg_points', 'standings_position']]
            y_driver = self.driver_features['predicted_points']

            self.driver_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.driver_model.fit(X_driver, y_driver)
            print("Using simplified driver model due to insufficient historical data")

        # Train constructor model
        if not constructor_train_df.empty:
            X_constructor_train = constructor_train_df[['price', 'recent_avg_points', 'recent_avg_position',
                                                        'circuit_avg_points', 'standings_position']]
            y_constructor_train = constructor_train_df['actual_points']

            # Split into train/test sets
            X_constructor_train, X_constructor_test, y_constructor_train, y_constructor_test = train_test_split(
                X_constructor_train, y_constructor_train, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_constructor_train_scaled = scaler.fit_transform(X_constructor_train)
            X_constructor_test_scaled = scaler.transform(X_constructor_test)

            # Train model with hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }

            grid_search = GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error'
            )

            grid_search.fit(X_constructor_train_scaled, y_constructor_train)

            # Get best model
            self.constructor_model = grid_search.best_estimator_

            # Evaluate model
            y_pred = self.constructor_model.predict(X_constructor_test_scaled)
            mse = mean_squared_error(y_constructor_test, y_pred)
            print(f"Constructor model MSE: {mse:.2f}, RMSE: {np.sqrt(mse):.2f}")

            # Prepare for prediction on current data
            self.constructor_scaler = scaler
        else:
            # Fallback to simpler model if no historical data
            X_constructor = self.constructor_features[['price', 'recent_avg_points', 'recent_avg_position',
                                                       'circuit_avg_points', 'standings_position']]
            y_constructor = self.constructor_features['predicted_points']

            self.constructor_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.constructor_model.fit(X_constructor, y_constructor)
            print("Using simplified constructor model due to insufficient historical data")

        print("Models trained successfully using historical race data")

    def predict_optimal_team(self):
        """Find the optimal team selection within budget constraints"""
        if self.driver_model is None or self.constructor_model is None:
            self.train_prediction_models()

        # Get predictions for all drivers and constructors
        driver_features = self.driver_features.copy()
        constructor_features = self.constructor_features.copy()

        # Use the trained model to predict points
        if hasattr(self, 'driver_scaler'):
            # If we used the advanced modeling approach
            X_driver = driver_features[['price', 'recent_avg_points', 'recent_avg_position',
                                        'recent_avg_grid', 'circuit_avg_points', 'standings_position']]
            X_driver_scaled = self.driver_scaler.transform(X_driver)
            driver_features['predicted_points'] = self.driver_model.predict(X_driver_scaled)
        else:
            # Fallback to simpler approach
            X_driver = driver_features[['price', 'recent_avg_points', 'recent_avg_position',
                                        'recent_avg_grid', 'circuit_avg_points', 'standings_position']]
            driver_features['predicted_points'] = self.driver_model.predict(X_driver)

        if hasattr(self, 'constructor_scaler'):
            # If we used the advanced modeling approach
            X_constructor = constructor_features[['price', 'recent_avg_points', 'recent_avg_position',
                                                  'circuit_avg_points', 'standings_position']]
            X_constructor_scaled = self.constructor_scaler.transform(X_constructor)
            constructor_features['predicted_points'] = self.constructor_model.predict(X_constructor_scaled)
        else:
            # Fallback to simpler approach
            X_constructor = constructor_features[['price', 'recent_avg_points', 'recent_avg_position',
                                                  'circuit_avg_points', 'standings_position']]
            constructor_features['predicted_points'] = self.constructor_model.predict(X_constructor)

        # Calculate value (points per million)
        driver_features['value'] = driver_features['predicted_points'] / driver_features['price']
        constructor_features['value'] = constructor_features['predicted_points'] / constructor_features['price']

        # Sort by predicted points (descending)
        drivers_sorted = driver_features.sort_values('predicted_points', ascending=False)
        constructors_sorted = constructor_features.sort_values('predicted_points', ascending=False)

        print(f"Top 5 drivers by predicted points:")
        for idx, (_, driver) in enumerate(drivers_sorted.head(5).iterrows()):
            print(
                f"{idx + 1}. {driver['full_name']} - {driver['predicted_points']:.2f} points, ${driver['price']}M, Value: {driver['value']:.2f}")

        print(f"\nTop 3 constructors by predicted points:")
        for idx, (_, constructor) in enumerate(constructors_sorted.head(3).iterrows()):
            print(
                f"{idx + 1}. {constructor['team_name']} - {constructor['predicted_points']:.2f} points, ${constructor['price']}M, Value: {constructor['value']:.2f}")

        # SIMPLIFIED BRUTE FORCE APPROACH TO AVOID PANDAS ISSUES
        print("\nEvaluating team combinations for optimal selection...")

        # Convert to Python lists to avoid pandas issues
        drivers_list = []
        for index, row in drivers_sorted.iterrows():
            driver_dict = {
                'full_name': row['full_name'],
                'team': row['team'],
                'price': row['price'],
                'predicted_points': row['predicted_points']
            }
            drivers_list.append(driver_dict)

        constructors_list = []
        for index, row in constructors_sorted.iterrows():
            constructor_dict = {
                'team_name': row['team_name'],
                'price': row['price'],
                'predicted_points': row['predicted_points']
            }
            constructors_list.append(constructor_dict)

        total_combinations = comb(len(drivers_list), 5) * comb(len(constructors_list), 2)
        print(f"Searching through approximately {total_combinations:,} potential team combinations")

        # Initialize variables for tracking best team
        optimal_team = None
        best_total_points = 0

        # Generate all valid combinations
        team_count = 0

        # For each possible driver combination (5 drivers)
        for driver_combo in itertools.combinations(drivers_list, 5):
            driver_cost = sum(d['price'] for d in driver_combo)
            driver_points = sum(d['predicted_points'] for d in driver_combo)

            # Only process constructor combinations if drivers are within budget
            if driver_cost < self.budget:
                # For each possible constructor combination (2 constructors)
                for c1_idx, constructor1 in enumerate(constructors_list):
                    for c2_idx in range(c1_idx + 1, len(constructors_list)):
                        constructor2 = constructors_list[c2_idx]

                        constructor_cost = constructor1['price'] + constructor2['price']
                        total_cost = driver_cost + constructor_cost

                        # Check if within budget
                        if total_cost <= self.budget:
                            team_count += 1
                            if team_count % 100000 == 0:
                                print(f"Evaluated {team_count:,} team combinations...")

                            # Calculate total points
                            constructor_points = constructor1['predicted_points'] + constructor2['predicted_points']
                            total_points = driver_points + constructor_points

                            # Check if better than current best
                            if total_points > best_total_points:
                                best_total_points = total_points
                                optimal_team = {
                                    'drivers': driver_combo,
                                    'constructors': [constructor1, constructor2],
                                    'total_cost': total_cost,
                                    'total_points': total_points
                                }

                                # Early termination if we have a good team with the top constructors
                                # This is an optimization - skip to next driver combo if we find a good one with top constructors
                                if c1_idx == 0 and c2_idx == 1:  # First and second constructors
                                    break  # Skip remaining constructors for this driver combo

        print(f"Completed evaluation. Analyzed {team_count:,} valid team combinations.")

        if optimal_team:
            print(
                f"Found optimal team with {optimal_team['total_points']:.2f} predicted points at ${optimal_team['total_cost']}M")
        else:
            print("Could not find a valid team within budget constraints")

        return optimal_team

    def predict_bonus_opportunities(self, track_name="Imola"):
        """Predict bonus opportunity targets (pole, podium, safety car, etc.)"""
        if self.driver_features is None or self.constructor_features is None:
            self.prepare_feature_engineering(track_name)

        print(f"\nAnalyzing bonus prediction opportunities for {track_name}...")

        # POLE POSITION PREDICTION
        # Find track circuit
        selected_circuit = self.find_circuit_by_name(track_name)
        track_id = selected_circuit['circuitId'] if selected_circuit is not None else None

        # Create a more sophisticated pole position prediction model
        pole_features = []

        for short_name, driver_info in self.driver_mapping.items():
            full_name = driver_info['full_name']
            team = driver_info['team']

            # Find driver in historical data
            driver_id = None
            for _, driver in self.drivers.iterrows():
                if f"{driver['forename']} {driver['surname']}".lower() == full_name.lower():
                    driver_id = driver['driverId']
                    break

            if driver_id is not None:
                # Get qualifying stats across all races
                driver_quali = self.qualifying[self.qualifying['driverId'] == driver_id]

                # Calculate pole rate (how often driver gets pole)
                total_quali = len(driver_quali)
                pole_count = len(driver_quali[driver_quali['position'] == 1])
                pole_rate = pole_count / total_quali if total_quali > 0 else 0

                # Average qualifying position
                avg_quali_position = driver_quali['position'].mean() if not driver_quali.empty else 15

                # First row performance (Q3 appearances)
                q3_count = len(driver_quali[driver_quali['q3'].notnull()])
                q3_rate = q3_count / total_quali if total_quali > 0 else 0

                # Track-specific qualifying performance
                track_quali = pd.DataFrame()

                if track_id is not None:
                    # Find all races at this track
                    track_races = self.races[self.races['circuitId'] == track_id]
                    track_race_ids = track_races['raceId'].tolist()

                    # Get qualifying results for these races
                    track_quali = driver_quali[driver_quali['raceId'].isin(track_race_ids)]

                # Calculate track-specific stats
                track_pole_count = len(track_quali[track_quali['position'] == 1]) if not track_quali.empty else 0
                track_avg_position = track_quali['position'].mean() if not track_quali.empty else 15

                # Get recent form (last 5 qualifying sessions)
                recent_quali = driver_quali.sort_values('qualifyId', ascending=False).head(5)
                recent_avg_position = recent_quali['position'].mean() if not recent_quali.empty else 15

                # Constructor factor - how good is the team at qualifying?
                constructor_id = None
                for _, constructor in self.constructors.iterrows():
                    if team.lower() in constructor['name'].lower():
                        constructor_id = constructor['constructorId']
                        break

                team_quali_advantage = 0
                if constructor_id is not None:
                    # Get team's qualifying performance
                    team_quali = self.qualifying[self.qualifying['constructorId'] == constructor_id]
                    team_avg_position = team_quali['position'].mean() if not team_quali.empty else 10

                    # Compare to overall grid average
                    overall_avg_position = self.qualifying['position'].mean()
                    team_quali_advantage = overall_avg_position - team_avg_position

                # Track type analysis for qualifying
                track_type_quali_factor = 1.0
                if selected_circuit is not None:
                    track_location = selected_circuit['location'].lower() if 'location' in selected_circuit else ''
                    track_name_lower = selected_circuit['name'].lower() if 'name' in selected_circuit else ''

                    # Drivers who excel at specific track types in qualifying
                    driver_skill_match = 1.0

                    # Street circuits/technical tracks
                    if any(x in track_name_lower or x in track_location for x in
                           ['monaco', 'singapore', 'baku', 'jeddah']):
                        if full_name in ['Charles Leclerc', 'Lando Norris', 'Fernando Alonso']:
                            driver_skill_match = 1.2

                    # High-speed circuits
                    elif any(x in track_name_lower or x in track_location for x in
                             ['monza', 'spa', 'silverstone']):
                        if full_name in ['Max Verstappen', 'Lewis Hamilton', 'George Russell']:
                            driver_skill_match = 1.15

                    # Technical circuits
                    elif any(x in track_name_lower or x in track_location for x in
                             ['hungary', 'barcelona', 'suzuka']):
                        if full_name in ['Fernando Alonso', 'Max Verstappen', 'Carlos Sainz']:
                            driver_skill_match = 1.1

                    # Apply car-track matching factor from driver features if available
                    if 'track_car_advantage' in self.driver_features.columns:
                        driver_row = self.driver_features[self.driver_features['full_name'] == full_name]
                        if not driver_row.empty:
                            track_type_quali_factor = driver_row['track_car_advantage'].values[0]

                    # Combine both factors
                    track_type_quali_factor = track_type_quali_factor * driver_skill_match

                # Compute pole score - higher is better
                pole_score = (
                                     (20 - avg_quali_position) * 0.25 +  # Better average position
                                     pole_rate * 40 +  # Higher pole frequency
                                     q3_rate * 15 +  # Q3 appearance rate
                                     (20 - recent_avg_position) * 0.4 +  # Recent form
                                     (20 - track_avg_position) * 0.3 +  # Track-specific performance
                                     team_quali_advantage * 5 +  # Team qualifying advantage
                                     track_pole_count * 10  # Previous poles at this track
                             ) * track_type_quali_factor  # Track type factor

                # Add to features
                pole_features.append({
                    'short_name': short_name,
                    'full_name': full_name,
                    'team': team,
                    'avg_quali_position': avg_quali_position,
                    'pole_rate': pole_rate,
                    'q3_rate': q3_rate,
                    'recent_avg_position': recent_avg_position,
                    'track_avg_position': track_avg_position,
                    'track_pole_count': track_pole_count,
                    'team_quali_advantage': team_quali_advantage,
                    'track_type_factor': track_type_quali_factor,
                    'pole_score': pole_score
                })

        # Convert to DataFrame and sort by pole score
        pole_df = pd.DataFrame(pole_features).sort_values('pole_score', ascending=False)

        # Top pole candidates
        top_pole_candidates = pole_df.head(3)

        print("\nTop 3 Pole Position Candidates:")
        for idx, (_, driver) in enumerate(top_pole_candidates.iterrows()):
            print(f"{idx + 1}. {driver['full_name']} ({driver['team']}) - Pole Score: {driver['pole_score']:.2f}")
            if idx == 0:
                print(
                    f"   Key factors: Pole Rate: {driver['pole_rate']:.2%}, Avg Quali: {driver['avg_quali_position']:.1f}, Track-Type Advantage: {driver['track_type_factor']:.2f}")

        # PODIUM PREDICTION
        # Use driver features and predicted points as basis
        podium_candidates = self.driver_features.sort_values('predicted_points', ascending=False).head(5)

        # SAFETY CAR PREDICTION
        # Analyze circuit characteristics and historical safety car deployment
        safety_car_probability = 0.5  # Default probability

        if track_id is not None:
            # Get historical races at this track
            track_races = self.races[self.races['circuitId'] == track_id]
            track_race_ids = track_races['raceId'].tolist()

            # Get results to analyze DNFs and incidents
            track_results = self.results[self.results['raceId'].isin(track_race_ids)]

            # Count DNFs (status codes indicating accident, mechanical failure, etc.)
            dnf_statuses = [1, 2, 3, 4, 11, 20, 29, 31, 41, 42, 43, 44, 45, 58, 81, 82, 104, 130,
                            137]  # Common DNF status codes
            dnf_count = track_results[track_results['statusId'].isin(dnf_statuses)].shape[0]

            # Average DNFs per race
            avg_dnfs_per_race = dnf_count / len(track_race_ids) if len(track_race_ids) > 0 else 0

            # Track-specific safety car likelihood
            is_challenging_circuit = False
            if selected_circuit is not None:
                track_location = selected_circuit['location'].lower() if 'location' in selected_circuit else ''
                track_name_lower = selected_circuit['name'].lower() if 'name' in selected_circuit else ''

                # Tracks known for high safety car probability
                high_sc_tracks = ['baku', 'monaco', 'singapore', 'jeddah', 'melbourne', 'monza', 'montreal']
                medium_sc_tracks = ['silverstone', 'spa', 'shanghai', 'austin', 'interlagos', 'imola']

                if any(track in track_name_lower or track in track_location for track in high_sc_tracks):
                    is_challenging_circuit = True
                    base_sc_probability = 0.75
                elif any(track in track_name_lower or track in track_location for track in medium_sc_tracks):
                    is_challenging_circuit = True
                    base_sc_probability = 0.6
                else:
                    base_sc_probability = 0.4

            # Calculate probability based on DNF rate and circuit characteristics
            if avg_dnfs_per_race > 4 or is_challenging_circuit:
                safety_car_probability = 0.75
            elif avg_dnfs_per_race > 2:
                safety_car_probability = 0.6
            else:
                safety_car_probability = 0.4

            print(f"\nSafety Car Analysis for {track_name}:")
            print(f"Historical races analyzed: {len(track_race_ids)}")
            print(f"Average DNFs per race: {avg_dnfs_per_race:.2f}")
            print(f"Probability of safety car deployment: {safety_car_probability:.2%}")

        # DRIVER 2x MULTIPLIER RECOMMENDATION
        # Look for best value (points/price) with consideration of risk
        driver_value = self.driver_features.copy()

        # Calculate risk factor based on historical DNF rate
        for idx, row in driver_value.iterrows():
            driver_name = row['full_name']
            driver_id = None

            # Find driver ID
            for _, driver in self.drivers.iterrows():
                if f"{driver['forename']} {driver['surname']}".lower() == driver_name.lower():
                    driver_id = driver['driverId']
                    break

            if driver_id is not None:
                # Calculate overall DNF rate
                driver_results = self.results[self.results['driverId'] == driver_id]
                dnf_statuses = [1, 2, 3, 4, 11, 20, 29, 31, 41, 42, 43, 44, 45, 58, 81, 82, 104, 130, 137]
                dnf_count = driver_results[driver_results['statusId'].isin(dnf_statuses)].shape[0]
                total_races = len(driver_results)
                dnf_rate = dnf_count / total_races if total_races > 0 else 0

                # Calculate track-specific DNF rate
                if track_id is not None:
                    track_races = self.races[self.races['circuitId'] == track_id]
                    track_race_ids = track_races['raceId'].tolist()
                    track_results = self.results[(self.results['driverId'] == driver_id) &
                                                 (self.results['raceId'].isin(track_race_ids))]

                    track_dnf_count = track_results[track_results['statusId'].isin(dnf_statuses)].shape[0]
                    track_total_races = len(track_results)
                    track_dnf_rate = track_dnf_count / track_total_races if track_total_races > 0 else dnf_rate

                    # Combine both rates (weighted more toward track-specific)
                    final_dnf_rate = track_dnf_rate * 0.6 + dnf_rate * 0.4
                else:
                    final_dnf_rate = dnf_rate

                # Add to DataFrame
                driver_value.loc[idx, 'dnf_rate'] = final_dnf_rate
            else:
                driver_value.loc[idx, 'dnf_rate'] = 0.2  # Default rate for new drivers

        # Calculate risk-adjusted value score
        driver_value['adjusted_value'] = driver_value['value'] * (1 - driver_value['dnf_rate'])

        # Get top value drivers
        top_value_drivers = driver_value.sort_values('adjusted_value', ascending=False).head(3)

        print("\nBest Driver Multiplier Candidates:")
        for idx, (_, driver) in enumerate(top_value_drivers.iterrows()):
            print(
                f"{idx + 1}. {driver['full_name']} - Value: {driver['value']:.2f}, DNF Risk: {driver['dnf_rate']:.2%}, Adjusted Value: {driver['adjusted_value']:.2f}")

        # Compile bonus predictions
        bonus_predictions = {
            'pole_prediction': top_pole_candidates.iloc[0]['full_name'] if not top_pole_candidates.empty else None,
            'podium_prediction': podium_candidates.head(3)[
                'full_name'].tolist() if not podium_candidates.empty else None,
            'safety_car_prediction': "Yes" if safety_car_probability > 0.5 else "No",
            'safety_car_probability': safety_car_probability,
            'driver_multiplier_recommendation': top_value_drivers.iloc[0][
                'full_name'] if not top_value_drivers.empty else None,
            'driver_multiplier_risk': top_value_drivers.iloc[0]['dnf_rate'] if not top_value_drivers.empty else None
        }

        return bonus_predictions

    def generate_team_report(self, optimal_team=None, bonus_predictions=None):
        """Generate comprehensive fantasy team report for Imola GP"""
        # If not provided, make predictions
        if optimal_team is None:
            optimal_team = self.predict_optimal_team()
        if bonus_predictions is None:
            bonus_predictions = self.predict_bonus_opportunities()

        # Create report
        print("\n" + "=" * 50)
        print("F1 FANTASY OPTIMAL TEAM REPORT - IMOLA GP")
        print("=" * 50 + "\n")

        print("SELECTED DRIVERS:")
        total_driver_cost = 0
        total_driver_points = 0
        for idx, driver in enumerate(optimal_team['drivers']):
            total_driver_cost += driver['price']
            total_driver_points += driver['predicted_points']
            print(
                f"{idx + 1}. {driver['full_name']} ({driver['team']}) - ${driver['price']}M - Predicted Points: {driver['predicted_points']:.2f}")

        print("\nSELECTED CONSTRUCTORS:")
        total_constructor_cost = 0
        total_constructor_points = 0
        for idx, constructor in enumerate(optimal_team['constructors']):
            total_constructor_cost += constructor['price']
            total_constructor_points += constructor['predicted_points']
            print(
                f"{idx + 1}. {constructor['team_name']} - ${constructor['price']}M - Predicted Points: {constructor['predicted_points']:.2f}")

        print("\nTEAM SUMMARY:")
        print(f"Total Cost: ${optimal_team['total_cost']}M / ${self.budget}M")
        print(f"Remaining Budget: ${self.budget - optimal_team['total_cost']}M")
        print(f"Expected Total Points: {optimal_team['total_points']:.2f}")

        print("\nBONUS PREDICTIONS:")
        print(f"Pole Position: {bonus_predictions['pole_prediction']} (+10 pts if correct)")
        print(f"Podium Prediction: {', '.join(bonus_predictions['podium_prediction'])} (+25 pts if correct)")
        print(
            f"Safety Car Prediction: {bonus_predictions['safety_car_prediction']} (Probability: {bonus_predictions['safety_car_probability']:.2f})")

        print(f"\nRECOMMENDED DRIVER MULTIPLIER: {bonus_predictions['driver_multiplier_recommendation']}")
        if 'driver_multiplier_risk' in bonus_predictions:
            print(f"DNF Risk: {bonus_predictions['driver_multiplier_risk']:.2%}")

        print("\nSTRATEGY NOTES:")
        print("1. Consider one driver swap before end of Q1 based on practice performance")
        print("2. Monitor practice sessions for any unexpected performance indicators")
        print("3. Check for any last-minute team upgrades or driver changes")

        return {
            "optimal_team": optimal_team,
            "bonus_predictions": bonus_predictions
        }


def run_fantasy_predictor(circuit_name=None):
    """
    Main function to run the F1 Fantasy predictor

    Parameters:
    -----------
    circuit_name : str
        Name of the circuit for which to make predictions (default: None, will prompt user)
    """
    calendar_2025 = [
        {"round": 1, "name": "Australian Grand Prix", "circuit": "Albert Park Circuit", "location": "Melbourne",
         "date": "March 16, 2025"},
        {"round": 2, "name": "Chinese Grand Prix", "circuit": "Shanghai International Circuit", "location": "Shanghai",
         "date": "March 23, 2025"},
        {"round": 3, "name": "Japanese Grand Prix", "circuit": "Suzuka International Racing Course",
         "location": "Suzuka", "date": "April 6, 2025"},
        {"round": 4, "name": "Bahrain Grand Prix", "circuit": "Bahrain International Circuit", "location": "Sakhir",
         "date": "April 13, 2025"},
        {"round": 5, "name": "Saudi Arabian Grand Prix", "circuit": "Jeddah Corniche Circuit", "location": "Jeddah",
         "date": "April 20, 2025"},
        {"round": 6, "name": "Miami Grand Prix", "circuit": "Miami International Autodrome", "location": "Miami",
         "date": "May 4, 2025"},
        {"round": 7, "name": "Emilia Romagna Grand Prix", "circuit": "Autodromo Enzo e Dino Ferrari",
         "location": "Imola", "date": "May 18, 2025"},
        {"round": 8, "name": "Monaco Grand Prix", "circuit": "Circuit de Monaco", "location": "Monte Carlo",
         "date": "May 25, 2025"},
        {"round": 9, "name": "Spanish Grand Prix", "circuit": "Circuit de Barcelona-Catalunya",
         "location": "Barcelona", "date": "June 1, 2025"},
        {"round": 10, "name": "Canadian Grand Prix", "circuit": "Circuit Gilles Villeneuve", "location": "Montreal",
         "date": "June 15, 2025"},
        {"round": 11, "name": "Austrian Grand Prix", "circuit": "Red Bull Ring", "location": "Spielberg",
         "date": "June 29, 2025"},
        {"round": 12, "name": "British Grand Prix", "circuit": "Silverstone Circuit", "location": "Silverstone",
         "date": "July 6, 2025"},
        {"round": 13, "name": "Belgian Grand Prix", "circuit": "Circuit de Spa-Francorchamps", "location": "Spa",
         "date": "July 27, 2025"},
        {"round": 14, "name": "Hungarian Grand Prix", "circuit": "Hungaroring", "location": "Budapest",
         "date": "August 3, 2025"},
        {"round": 15, "name": "Dutch Grand Prix", "circuit": "Circuit Zandvoort", "location": "Zandvoort",
         "date": "August 31, 2025"},
        {"round": 16, "name": "Italian Grand Prix", "circuit": "Autodromo Nazionale Monza", "location": "Monza",
         "date": "September 7, 2025"},
        {"round": 17, "name": "Azerbaijan Grand Prix", "circuit": "Baku City Circuit", "location": "Baku",
         "date": "September 21, 2025"},
        {"round": 18, "name": "Singapore Grand Prix", "circuit": "Marina Bay Street Circuit", "location": "Singapore",
         "date": "October 5, 2025"},
        {"round": 19, "name": "United States Grand Prix", "circuit": "Circuit of the Americas", "location": "Austin",
         "date": "October 19, 2025"},
        {"round": 20, "name": "Mexico City Grand Prix", "circuit": "Autódromo Hermanos Rodríguez",
         "location": "Mexico City", "date": "October 26, 2025"},
        {"round": 21, "name": "São Paulo Grand Prix", "circuit": "Autódromo José Carlos Pace", "location": "São Paulo",
         "date": "November 9, 2025"},
        {"round": 22, "name": "Las Vegas Grand Prix", "circuit": "Las Vegas Strip Circuit", "location": "Las Vegas",
         "date": "November 22, 2025"},
        {"round": 23, "name": "Qatar Grand Prix", "circuit": "Losail International Circuit", "location": "Lusail",
         "date": "November 30, 2025"},
        {"round": 24, "name": "Abu Dhabi Grand Prix", "circuit": "Yas Marina Circuit", "location": "Abu Dhabi",
         "date": "December 7, 2025"}
    ]

    # Get available circuits from database if possible
    try:
        circuits_df = pd.read_csv('./circuits.csv')
        available_circuits = sorted(list(set([circuit['name'] for _, circuit in circuits_df.iterrows()])))
    except Exception:
        # Use hardcoded list of major circuits if we can't read the circuit file
        available_circuits = [race["circuit"] for race in calendar_2025]

    # Let user select a circuit if not provided
    if circuit_name is None:
        print("\n2025 F1 CALENDAR")
        print("=" * 80)
        print(f"{'Round':<7}{'Grand Prix':<30}{'Circuit':<35}{'Date':<15}")
        print("-" * 80)
        for race in calendar_2025:
            print(f"{race['round']:<7}{race['name']:<30}{race['circuit']:<35}{race['date']:<15}")
        print("=" * 80)

        while True:
            try:
                choice = input("\nEnter race number, circuit name, or 'next' for the upcoming race: ")

                if choice.lower() == 'next':
                    # Find the next race based on today's date
                    from datetime import datetime
                    current_date = datetime.now()

                    # Find next race after current date
                    next_race = None
                    for race in calendar_2025:
                        race_date = datetime.strptime(race["date"], "%B %d, %Y")
                        if race_date > current_date:
                            next_race = race
                            break

                    if next_race:
                        circuit_name = next_race["circuit"]
                        print(f"Selected upcoming race: {next_race['name']} at {circuit_name} on {next_race['date']}")
                        break
                    else:
                        print("No upcoming races found in the 2025 calendar.")
                        circuit_name = calendar_2025[0]["circuit"]  # Default to first race
                        break

                # Check if input is a round number
                if choice.isdigit():
                    round_num = int(choice)
                    if 1 <= round_num <= len(calendar_2025):
                        selected_race = calendar_2025[round_num - 1]
                        circuit_name = selected_race["circuit"]
                        print(f"Selected: {selected_race['name']} at {circuit_name}")
                        break
                    else:
                        print(f"Invalid round number. Please enter a number between 1 and {len(calendar_2025)}")
                else:
                    # Try to find a match in the grand prix or circuit names
                    matches = []
                    for race in calendar_2025:
                        if (choice.lower() in race["name"].lower() or
                                choice.lower() in race["circuit"].lower() or
                                choice.lower() in race["location"].lower()):
                            matches.append(race)

                    if len(matches) == 1:
                        circuit_name = matches[0]["circuit"]
                        print(f"Selected: {matches[0]['name']} at {circuit_name}")
                        break
                    elif len(matches) > 1:
                        print("Multiple matches found. Please select one:")
                        for i, match in enumerate(matches):
                            print(f"{i + 1}. {match['name']} at {match['circuit']}")
                        sub_choice = input("Enter the number of your choice: ")
                        if sub_choice.isdigit() and 0 < int(sub_choice) <= len(matches):
                            circuit_name = matches[int(sub_choice) - 1]["circuit"]
                            print(f"Selected: {matches[int(sub_choice) - 1]['name']} at {circuit_name}")
                            break
                    else:
                        print("No matching race found. Please try again or enter a round number.")
            except Exception as e:
                print(f"Error: {e}. Please try again.")

    # Use Imola as default if still no valid circuit
    if circuit_name is None:
        circuit_name = "Autodromo Enzo e Dino Ferrari"  # Imola
        print(f"Using default circuit: {circuit_name}")

    print(f"\n{'=' * 70}")
    print(f"F1 FANTASY PREDICTOR - {circuit_name.upper()}")
    print(f"{'=' * 70}\n")

    # Initialize the predictor
    predictor = F1FantasyPredictor(data_path='./', budget=100)

    # Load data
    print("Step 1: Loading historical F1 data...")
    load_success = predictor.load_data()
    if not load_success:
        print("Failed to load data, exiting")
        return

    # Map current names to historical data
    print("\nStep 2: Mapping current drivers/teams to historical data...")
    predictor.map_current_names()

    # Prepare features
    print(f"\nStep 3: Engineering features for prediction models for {circuit_name}...")
    driver_features, constructor_features = predictor.prepare_feature_engineering(circuit_name)

    # Train machine learning models using historical data
    print("\nStep 4: Training prediction models on historical race data...")
    predictor.train_prediction_models()

    # Find optimal team
    print("\nStep 5: Finding optimal team combination...")
    optimal_team = predictor.predict_optimal_team()

    # Predict bonus opportunities
    print("\nStep 6: Analyzing bonus prediction opportunities...")
    bonus_predictions = predictor.predict_bonus_opportunities(circuit_name)

    # Generate comprehensive report
    print("\nStep 7: Generating final fantasy team report...\n")

    print("=" * 70)
    print(f"OPTIMAL F1 FANTASY TEAM - {circuit_name}")
    print("=" * 70)

    print("\n🏎️ SELECTED DRIVERS:")
    total_driver_cost = 0
    total_driver_points = 0
    for idx, driver in enumerate(optimal_team['drivers']):
        total_driver_cost += driver['price']
        total_driver_points += driver['predicted_points']
        print(f"{idx + 1}. {driver['full_name']} ({driver['team']}) - ${driver['price']}M")
        print(f"   Predicted Points: {driver['predicted_points']:.2f}")

    print("\n🏭 SELECTED CONSTRUCTORS:")
    total_constructor_cost = 0
    total_constructor_points = 0
    for idx, constructor in enumerate(optimal_team['constructors']):
        total_constructor_cost += constructor['price']
        total_constructor_points += constructor['predicted_points']
        print(f"{idx + 1}. {constructor['team_name']} - ${constructor['price']}M")
        print(f"   Predicted Points: {constructor['predicted_points']:.2f}")

    print("\n💰 TEAM SUMMARY:")
    print(f"Total Cost: ${optimal_team['total_cost']}M / ${predictor.budget}M")
    print(f"Remaining Budget: ${predictor.budget - optimal_team['total_cost']}M")
    print(f"Expected Total Points: {optimal_team['total_points']:.2f}")

    print("\n🎯 BONUS PREDICTIONS:")
    print(f"Pole Position: {bonus_predictions['pole_prediction']} (+10 pts if correct)")
    print(f"Podium Prediction: {', '.join(bonus_predictions['podium_prediction'])} (+25 pts if correct)")
    print(
        f"Safety Car Prediction: {bonus_predictions['safety_car_prediction']} (Probability: {bonus_predictions['safety_car_probability']:.2%})")

    print(f"\n⭐ RECOMMENDED DRIVER MULTIPLIER: {bonus_predictions['driver_multiplier_recommendation']}")
    if 'driver_multiplier_risk' in bonus_predictions:
        print(f"   DNF Risk: {bonus_predictions['driver_multiplier_risk']:.2%}")

    print("\n📋 STRATEGY NOTES:")
    print("1. Consider one driver swap before end of Q1 based on practice performance")
    print("2. Monitor practice sessions for any unexpected performance indicators")
    print("3. Check for any last-minute team upgrades or driver changes")
    print("4. Look for track-specific advantages that may shift performance expectations")

    print(f"\n{'=' * 70}")
    print("MODEL INFORMATION & METHODOLOGY")
    print(f"{'=' * 70}")
    print("This prediction model uses historical F1 data to optimize team selection:")
    print(f"- Circuit-specific performance analysis at {circuit_name}")
    print("- Recent form evaluation (weighted more heavily than historical data)")
    print("- Car performance analysis (with track-specific adjustments)")
    print("- Machine learning models trained on multiple seasons of race data")
    print("- Value optimization (points per million) within budget constraints")
    print("- Consideration of DNF risk for multiplier recommendations")
    print("- Analysis of qualifying performance patterns for pole predictions")
    print(f"{'=' * 70}")

    return {
        "optimal_team": optimal_team,
        "bonus_predictions": bonus_predictions
    }


if __name__ == "__main__":
    run_fantasy_predictor()  # Will prompt user for circuit selection using 2025 calendar
