import os
import requests

import dotenv

dotenv.load_dotenv('./../../../')

def end_session():
  RUNPOD_URL = os.environ.get('RUNPOD_URL')
  RUNPOD_POD_ID = os.environ.get('RUNPOD_POD_ID')
  RUNPOD_BEARER_TOKEN = os.environ.get('RUNPOD_BEARER_TOKENARER_TOKEN')
  
  response = requests.get(
    f'{RUNPOD_URL}/pods/{RUNPOD_POD_ID}/stop',
    headers={
      'Authorization': f'Bearer {RUNPOD_BEARER_TOKEN}'
    }
  )
  
  return response