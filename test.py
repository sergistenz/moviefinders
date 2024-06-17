import requests
import os
from dotenv import load_dotenv
import pandas as pd
from moviefinder import genres_list
from moviefinder import get_movie_dataset


get_movie_dataset()