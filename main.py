import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import itertools
import warnings

warnings.filterwarnings('ignore')


class F1FantasyPredictor:
    def __init__(self, data_path='./', budget=100):
        self.data_path = data_path
        self.budget = budget

        # Store pricing for constructors and drivers
        self.constructor_pricing = {
            "Mclaren": 30,
            "Mercedes": 25,
            "Ferrari": 24,
            "Redbull": 23,
            "Williams": 20,
            "Racing Bulls": 16,
            "Haas": 15,
            "Alpine": 14,
            "Aston Martin": 13,
            "Stake": 12
        }

        self.driver_pricing = {
            "Oscar": 30,
            "Lando": 25,
            "Max": 24,
            "George": 23,
            "Kimi": 22,
            "Charles": 21,
            "Albon": 20,
            "Yuki": 16,
            "Isack": 15,
            "Lewis": 14,
            "Pierre": 13,
            "Carlos": 12,
            "Bearman": 11,
            "Ocon": 10,
            "Alonso": 9,
            "Colapinto": 8.5,
            "Stroll": 7,
            "Lawson" : 6.5,
            "Hulk": 6,
            "Bortoleto": 5.5
        }

        # Map the drivers to their full names and teams (2025 season assumption)
        self.driver_mapping = {
            "Oscar": {"full_name": "Oscar Piastri", "team": "Mclaren"},
            "Lando": {"full_name": "Lando Norris", "team": "Mclaren"},
            "Max": {"full_name": "Max Verstappen", "team": "Redbull"},
            "George": {"full_name": "George Russell", "team": "Mercedes"},
            "Kimi": {"full_name": "Kimi Antonelli", "team": "Mercedes"},
            "Charles": {"full_name": "Charles Leclerc", "team": "Ferrari"},
            "Albon": {"full_name": "Alex Albon", "team": "Williams"},
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
            "Hulk": {"full_name": "Nico Hulkenberg", "team": "Stake"},
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

    def prepare_feature_engineering(self):
        """Engineer features for prediction model"""
        # Get circuit performance at Imola
        imola_circuit = self.find_circuit_by_name("Imola")
        if imola_circuit is None:
            print("Imola circuit not found, using generic model")
            imola_performance = pd.DataFrame()
        else:
            imola_performance = self.get_circuit_historical_performance(imola_circuit['circuitId'])

        # Get recent form
        driver_form, constructor_form = self.analyze_recent_form(n_races=10)

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

            # Get Imola performance
            imola_stats = imola_performance[
                imola_performance['driverId'] == driver_id] if driver_id and not imola_performance.empty else None

            # Calculate value metrics
            recent_avg_points = recent_stats['points'].values[
                0] if recent_stats is not None and not recent_stats.empty else 0
            recent_avg_position = recent_stats['positionOrder'].values[
                0] if recent_stats is not None and not recent_stats.empty else 20
            recent_avg_grid = recent_stats['grid'].values[
                0] if recent_stats is not None and not recent_stats.empty else 20

            imola_avg_points = imola_stats['points'].mean() if imola_stats is not None and not imola_stats.empty else 0

            # Get standings position
            if driver_id:
                latest_standing = self.driver_standings[self.driver_standings['driverId'] == driver_id].sort_values(
                    'raceId', ascending=False).head(1)
                standings_position = latest_standing['position'].values[0] if not latest_standing.empty else 20
            else:
                standings_position = 20

            # Calculate predicted points (basic formula, will be replaced by ML model)
            predicted_points = (recent_avg_points * 0.6 + imola_avg_points * 0.4)

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
                'imola_avg_points': imola_avg_points,
                'standings_position': standings_position,
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

            # Get Imola performance
            imola_team_stats = imola_performance[imola_performance[
                                                     'constructorId'] == constructor_id] if constructor_id and not imola_performance.empty else None

            # Calculate value metrics
            recent_avg_points = recent_stats['points'].values[
                0] if recent_stats is not None and not recent_stats.empty else 0
            recent_avg_position = recent_stats['positionOrder'].values[
                0] if recent_stats is not None and not recent_stats.empty else 10

            imola_avg_points = imola_team_stats[
                'points'].mean() if imola_team_stats is not None and not imola_team_stats.empty else 0

            # Get standings position
            if constructor_id:
                latest_standing = self.constructor_standings[
                    self.constructor_standings['constructorId'] == constructor_id].sort_values('raceId',
                                                                                               ascending=False).head(1)
                standings_position = latest_standing['position'].values[0] if not latest_standing.empty else 10
            else:
                standings_position = 10

            # Calculate predicted points (basic formula, will be replaced by ML model)
            predicted_points = (
                                           recent_avg_points * 0.6 + imola_avg_points * 0.4) * 2  # Constructors typically get double points

            # Calculate value (points per million)
            value = predicted_points / price if price > 0 else 0

            # Store features
            constructor_features.append({
                'team_name': team_name,
                'price': price,
                'recent_avg_points': recent_avg_points,
                'recent_avg_position': recent_avg_position,
                'imola_avg_points': imola_avg_points,
                'standings_position': standings_position,
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

        # For drivers
        X_driver = self.driver_features[['price', 'recent_avg_points', 'recent_avg_position',
                                         'recent_avg_grid', 'imola_avg_points', 'standings_position']]
        y_driver = self.driver_features['predicted_points']

        # For constructors
        X_constructor = self.constructor_features[['price', 'recent_avg_points', 'recent_avg_position',
                                                   'imola_avg_points', 'standings_position']]
        y_constructor = self.constructor_features['predicted_points']

        # In a real scenario, we would train models on historical data
        # For now, we'll use a simple model that returns our handcrafted predictions

        # For demonstration, we'll use a simple RandomForest that just memorizes our data
        self.driver_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.driver_model.fit(X_driver, y_driver)

        self.constructor_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.constructor_model.fit(X_constructor, y_constructor)

        print("Models trained successfully")

    def predict_optimal_team(self):
        """Find the optimal team selection within budget constraints"""
        if self.driver_model is None or self.constructor_model is None:
            self.train_prediction_models()

        # Get predictions for all drivers and constructors
        self.driver_features['predicted_points'] = self.driver_model.predict(
            self.driver_features[['price', 'recent_avg_points', 'recent_avg_position',
                                  'recent_avg_grid', 'imola_avg_points', 'standings_position']]
        )

        self.constructor_features['predicted_points'] = self.constructor_model.predict(
            self.constructor_features[['price', 'recent_avg_points', 'recent_avg_position',
                                       'imola_avg_points', 'standings_position']]
        )

        # Sort by predicted points (descending)
        drivers_sorted = self.driver_features.sort_values('predicted_points', ascending=False)
        constructors_sorted = self.constructor_features.sort_values('predicted_points', ascending=False)

        # Calculate team combinations within budget
        optimal_team = None
        best_total_points = 0

        # Generate all combinations of 5 drivers
        for drivers_combo in itertools.combinations(drivers_sorted.iterrows(), 5):
            selected_drivers = [driver[1] for driver in drivers_combo]
            driver_cost = sum(driver['price'] for driver in selected_drivers)
            driver_points = sum(driver['predicted_points'] for driver in selected_drivers)

            # Only consider constructor combinations if we have enough budget left
            remaining_budget = self.budget - driver_cost
            if remaining_budget >= constructors_sorted.iloc[-1][
                'price'] * 2:  # At least enough for 2 cheapest constructors

                # Generate all combinations of 2 constructors
                for constructors_combo in itertools.combinations(constructors_sorted.iterrows(), 2):
                    selected_constructors = [constructor[1] for constructor in constructors_combo]
                    constructor_cost = sum(constructor['price'] for constructor in selected_constructors)
                    constructor_points = sum(constructor['predicted_points'] for constructor in selected_constructors)

                    total_cost = driver_cost + constructor_cost
                    total_points = driver_points + constructor_points

                    # Check if team is within budget and better than current best
                    if total_cost <= self.budget and total_points > best_total_points:
                        best_total_points = total_points
                        optimal_team = {
                            'drivers': selected_drivers,
                            'constructors': selected_constructors,
                            'total_cost': total_cost,
                            'total_points': total_points
                        }

        return optimal_team

    def predict_bonus_opportunities(self):
        """Predict bonus opportunity targets (pole, podium, safety car, etc.)"""
        if self.driver_features is None or self.constructor_features is None:
            self.prepare_feature_engineering()

        # For pole prediction
        # Sort drivers by qualifying performance
        qualified_drivers = self.driver_features.copy()
        qualified_drivers['quali_score'] = qualified_drivers['recent_avg_grid'] + (
                    20 - qualified_drivers['recent_avg_position']) * 0.5
        pole_candidates = qualified_drivers.sort_values('quali_score').head(3)

        # For podium prediction
        podium_candidates = self.driver_features.sort_values('predicted_points', ascending=False).head(5)

        # For safety car prediction (simplified logic)
        imola_circuit = self.find_circuit_by_name("Imola")
        safety_car_probability = 0.7  # Default probability

        if imola_circuit is not None:
            # Check historical races at Imola for safety car appearances
            imola_races = self.races[self.races['circuitId'] == imola_circuit['circuitId']]

            # Note: In a real implementation, we would need data about safety car deployments,
            # which isn't directly in our dataset. This would be supplemented with external data.

            # For demonstration, we'll use race statistics as a proxy
            imola_race_ids = imola_races['raceId'].tolist()
            imola_results = self.results[self.results['raceId'].isin(imola_race_ids)]

            # Check for retirements (status code > 0 indicates non-finish)
            retirement_count = imola_results[imola_results['statusId'] > 0].shape[0]
            avg_retirements_per_race = retirement_count / len(imola_race_ids) if len(imola_race_ids) > 0 else 0

            # Higher retirement rates suggest higher safety car probability
            if avg_retirements_per_race > 5:
                safety_car_probability = 0.8
            elif avg_retirements_per_race > 3:
                safety_car_probability = 0.65
            else:
                safety_car_probability = 0.4

        # Compile bonus predictions
        bonus_predictions = {
            'pole_prediction': pole_candidates.iloc[0]['full_name'] if not pole_candidates.empty else None,
            'podium_prediction': podium_candidates.head(3)[
                'full_name'].tolist() if not podium_candidates.empty else None,
            'safety_car_prediction': "Yes" if safety_car_probability > 0.5 else "No",
            'safety_car_probability': safety_car_probability,
            'driver_multiplier_recommendation': self.driver_features.sort_values('value', ascending=False).iloc[0][
                'full_name']
        }

        return bonus_predictions

    def generate_team_report(self):
        """Generate comprehensive fantasy team report for Imola GP"""
        # Make predictions
        optimal_team = self.predict_optimal_team()
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

        print("\nSTRATEGY NOTES:")
        print("1. Consider one driver swap before end of Q1 based on practice performance")
        print("2. Monitor practice sessions for any unexpected performance indicators")
        print("3. Check for any last-minute team upgrades or driver changes")

        return {
            "optimal_team": optimal_team,
            "bonus_predictions": bonus_predictions
        }


# Main execution function
def run_fantasy_predictor(circuit_name="Imola"):
    # Initialize the predictor
    predictor = F1FantasyPredictor(data_path='./', budget=100)

    # Load data
    load_success = predictor.load_data()
    if not load_success:
        print("Failed to load data, exiting")
        return

    # Map current names to historical data
    predictor.map_current_names()

    # Prepare features
    driver_features, constructor_features = predictor.prepare_feature_engineering()

    # Train models
    predictor.train_prediction_models()

    # Generate report
    predictor.generate_team_report()

    # Output final recommendations
    print("\nFINAL RECOMMENDATIONS:")
    print("1. Submit your team according to the optimal selection above")
    print("2. Save one driver change for after practice sessions")
    print("3. Use your 2x multiplier on the recommended driver with highest value")
    print("4. Consider the bonus predictions for additional points")


if __name__ == "__main__":
    run_fantasy_predictor(circuit_name="Imola")