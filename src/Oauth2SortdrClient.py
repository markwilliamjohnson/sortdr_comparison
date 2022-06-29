import os
from time import time
import requests

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# # This allows us to use a plain HTTP callback
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = "1"

client = BackendApplicationClient(client_id=os.environ['CLIENT_ID'])
oauth = OAuth2Session(client=client)


def get_access_token():
    # Fetch an access token from the provider
    access_token = ''
    try:
        access_token = oauth.fetch_token(token_url=os.environ['TOKEN_ENDPOINT'], client_id=os.environ['CLIENT_ID'],
                             client_secret=os.environ['CLIENT_SECRET'])
    except Exception as err:
        print('ERROR: could not get an access token: ', err) 
    return access_token

token = get_access_token()

def is_token_valid():
    if token is None:
        return False
    token_expiry_time = token["expires_at"]
    token_expiry = token_expiry_time - time()
    if token_expiry <= 10:
        return False
    return True

def get_token():
    global token
    if not is_token_valid():
        token = get_access_token()
    return token

def call_sortdr_api(endpoint):
    token = get_token()
    headers = {"Authorization": "bearer " + token['access_token']}
    return requests.get(endpoint, headers=headers)