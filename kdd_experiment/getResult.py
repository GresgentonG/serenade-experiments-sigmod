import requests
from requests.exceptions import HTTPError
try:
    myurl = 'http://localhost:8080/v1/recommend'
    params = dict(
        session_id='14',
        user_consent='true',
        # item_id='453279',
        # item_id='72916',
        item_id="360017",
    )
    response = requests.get(url=myurl, params=params)
    response.raise_for_status()
    # access json content
    jsonResponse = response.json()
    print(jsonResponse)
except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')
except Exception as err:
    print(f'Other error occurred: {err}')