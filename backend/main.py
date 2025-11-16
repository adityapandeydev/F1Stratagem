import fastf1
from fastf1 import plotting
import pandas as pd

fastf1.Cache.enable_cache('cache')

def get_team_race_pace(year: int, team: str):
    race_weekends = fastf1.get_event_schedule(year)