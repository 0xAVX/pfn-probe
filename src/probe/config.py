"""Shared task config: costs, free features, categoricals, Bayes decision."""
from __future__ import annotations

COSTS = {"Age": 0, "Gender": 0, "City_Type": 0, "Daily_Commute_km": 1,
         "Number_of_Cars_Owned": 1, "Environmental_Concern_Level": 1,
         "Charging_Stations_Near_Home": 2, "Charging_Stations_Near_Work": 2,
         "Current_Car_Type": 2, "Range_Anxiety_Level": 2,
         "Home_Charging_Possible": 3, "Subsidy_Available": 3,
         "Annual_Income_USD": 5}
FREE = ["Age", "Gender", "City_Type"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]


def decide(p, c_fn=5.0, c_fp=1.0):
    return int(p * c_fn > (1 - p) * c_fp)
