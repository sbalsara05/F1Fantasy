import pandas as pd
import numpy as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import itertools
import warnings

warnings.filterwarnings("ignore")


# Class for Fantasy Predictor
class F1FantasyPredictor:
    def __init__(self, data_path='./', budget=100):
        self.data_path = data_path
        self.budget = budget

        # Store Pricing for constructors and drivers
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
            "Sauber": 12
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
            "Stroll": 8.5,
            "Lawson": 7,
            "Hulk": 6.5,
            "Bortoleto": 6,
            "Doohan": 5.5
        }

        # Map the drivers to their respective teams
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
            "Doohan": {"full_name": "Jack Doohan", "team": "Alpine"}
        }

        # Initailize empty data frames
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

        # Strore the predition models
        self.driver_model = None
        self.constructor_model = None

        # Store processed features for predictions
        self.driver_features = None
        self.constructor_features = None

        # Function to load all the data from the cs files

    def load_data(self):
        try:
            self.races = pd.read_csv("{0}races.csv".format(self.data_path))
            self.drivers = pd.read_csv("{0}drivers.csv".format(self.data_path))
            self.constructors = pd.read_csv("{0}constructors.csv".format(self.data_path))
            self.results = pd.read_csv("{0}results.csv".format(self.data_path))
            self.qualifying = pd.read_csv("{0}qualifying.csv".format(self.data_path))
            self.circuits = pd.read_csv("{0}circuits.csv".format(self.data_path))
            self.constructor_results = pd.read_csv("{0}constructor_results.csv".format(self.data_path))
            self.constructor_standings = pd.read_csv("{0}constructor_standings".format(self.data_path))
            self.driver_standings = pd.read_csv("{0}driver_standings".format(self.data_path))
            self.lap_times = pd.read_csv("{0}.lap_times".format(self.data_path))
            self.pit_stops = pd.read_csv("{0}pit_stops.csv".format(self.data_path))

            print("Data loaded Successfully!")
            return True
        except Exception as e:
            print("Error loading {0}".format(e))
            return False

    # Map historical driver and constructor data to current names
    def map_current_names(self):
        self.driver_name_to_id = {}
        for driver_id, row in self.drivers.iterrows():
            full_name = f"{row["forename"]} {row["surname"]}"
            self.driver_name_to_id[full_name.lower()] = row["driver_id"]

        # Map current drivers/constructors to historical IDs
        self.current_driver_ids = {}
        for short_name, info in self.races.iterrows():
            full_name = info["full_name"].lower()
            if full_name in self.current_driver_ids:
                self.current_driver_ids[short_name] = self.driver_name_to_id[full_name]
            else:
                print("Warning: No driver found for {0}".format(info["full_name"]))
                self.current_driver_ids = None

        # Create constuctor mapping
        self.current_constructor_ids = {}
        for constructor_ref, row in self.constructors.iterrows():
            for team_name in self.constructor_pricing.keys():
                # Check if constcutor name matches
                if team_name.lower() in row['name'].lower():
                    self.current_constructor_ids[team_name] = row['constructorId']
                    break

                # Find circuit by name

    def find_circuit_by_name(self, circuit_name="Imola"):
        for i, circuit in self.circuits.iterrows():
            if circuit_name.lower() in circuit['name'].lower() or circuit_name.lower() in circuit[
                'location'].lower():
                return circuit
            return None

            # Get the historical circuit performance

    def get_circuit_historical_performance(self, circuit_id):
        circuit_races = self.races[self.races['circuitId'] == circuit_id]

        race_ids = circuit_races['raceId'].tolist()

        # Get all the results from these circuits
        circuit_results = self.results[self.results['raceId'].isin(race_ids)]

        # Join the driver and constructor data
        circuit_results = circuit_results.merge(self.drivers[["driver_id", "forename", "surname"]],
                                                        on="driver_id")
        circuit_results = circuit_results.merge(self.constructors[["constructorId", "name"]],
                                                        on="constructorId")

        # Add the race year
        circuit_results = circuit_results.merge(self.races[["raceId", "year"]], on="raceId")

        # Sort by the year (DESC) and position (ASC)
        circuit_results = circuit_results.sort_values(by=["year", "positionOrder"], ascending=[False, True])

        # Analyze the recent forms of the drivers (5 races, increase later)

    def analyze_recent_form(self, n_races=5):
        recent_races = self.races.sort_values(by="date", ascending=False).head(n_races)
        race_ids = recent_races['raceId'].tolist()

        # Get results from the most recent races
        recent_results = self.results[self.results['raceId'].isin(race_ids)]

