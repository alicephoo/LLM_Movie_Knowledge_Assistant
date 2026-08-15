import os
import atexit

from dotenv import load_dotenv
from openai import OpenAI

from src.monitoring import create_logs_table


create_logs_table()

print("Monitoring table created successfully.")