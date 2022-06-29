import datetime
import os
import sys

import requests
from flask import request, json

import Oauth2SortdrClient as api_service
import numpy as np


def get_output(predictions,referralThreshold,reference_data):
    for i in range(len(reference_data)):
        reference_data[i]['multiplePredictedValues'] = []
        reference_data[i]['multipleProbabilityValues'] = []

    for prediction in predictions:
        comparisons_result = np.array(prediction['comparisons_result'])
        predicted_values = list(np.argmax(comparisons_result, axis=-1))
        probability_values = [index[1] for index in comparisons_result]
        #print(predicted_values)
        #print(probability_values)
        for j in range(len(predicted_values)):
            reference_data[j]['multiplePredictedValues'].append(int(predicted_values[j]))
            reference_data[j]['multipleProbabilityValues'].append(float("{:.3f}".format(probability_values[j])))

    # call the stats service to amalgmate all comparison results into a single result
    stats_response = amalgamate_comparison_results(predictions, referralThreshold,reference_data)
    stats_output = {}
    if (bool(stats_response)):
        confidence_interval = {}
        stats_output['recommendation'] = stats_response['Recommendation']
        stats_output['probability for having referrable DR'] = stats_response['Probability for having referrable DR']
        confidence_interval['lower'] = stats_response['Confidence interval']['Prob_lower']
        confidence_interval['upper'] = stats_response['Confidence interval']['Prob_upper']
        stats_output['confidence interval'] = confidence_interval

        probability_values = stats_response['Amalgamated comparison result']['prob_vals']
        predicted_values = stats_response['Amalgamated comparison result']['predicted_vals']
        flagged_imgs = stats_response['Flagged image ids for ACJ']

        rank = compute_rank_range(np.asarray(predicted_values))
        if len(rank) > 0:
            min_range = rank[0]
            max_range = rank[len(rank) - 1]
            rank_range = {"min": min_range, "max": max_range}
            #print(rank_range)

        for i in range(len(reference_data)):
            reference_data[i]['predictedValue'] = predicted_values[i]
            reference_data[i]['probabilityValue'] = probability_values[i]
            reference_data[i]['show'] = flagged_imgs[i] == 1

    return {'rank': rank_range, 'ComparisonResult': reference_data, "stats_ouput": stats_output}

def get_predictions(test_image):
    payload = json.dumps({"image": str(test_image, 'utf-8')})
    url = os.environ["MODELS_SERVER_ENDPOINT"]
    headers = {"content-type": "application/json"}

    print("INFO: getting all predictions")
    start_time = datetime.datetime.now()
    response = {}
    try:
        response = requests.post(url, data=payload, headers=headers)
    except requests.exceptions.RequestException as e: 
        print("ERROR: AI Models Service Connection Failure")
        print(e)
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    print("INFO: time taken for all predictions: ", str(time_taken))

    if response.status_code != 200:
        print("ERROR: AI Models Service - could not get predictions from models server")
        print("ERROR: AI Models Service Response code: " + str(response.status_code))
        return []
    output = response.json()
    return output['predictions']

def amalgamate_comparison_results(predictions,referralThreshold,reference_data):
    input = {}
    input_models = []
    input_images = []

    for i in range(len(reference_data)):
        id=reference_data[i]['id']
        sortdr_classification=reference_data[i]['classifications'].split(",")[0]
        weighted_score=reference_data[i]['weightedScore']
        input_images.append({"id": id, "classification": sortdr_classification, "score":weighted_score})

    for i in range(len(predictions)):
        # pred_list = list(np.argmax(predictions[i], axis=-1)) # returns the indices of the maximum values along an axis
        # probability_list = list(np.max(predictions[i], axis=-1))  # returns max value in each array
        comparisons_result = predictions[i]['comparisons_result']
        model_name = predictions[i]['name']
        pred_list = list(np.argmax(comparisons_result, axis=-1))
        probability_list = [index[1] for index in comparisons_result]
        pred_vals=[]
        prob_vals = []
        for j in range(len(pred_list)):
            pred_vals.append(int(pred_list[j]))
            prob_vals.append(float(probability_list[j]))
        input_models.append({"name": model_name, "prob_vals": prob_vals, "predicted_vals": pred_vals})

    input['Models'] = input_models
    input['Images'] = input_images
    input['Threshold'] = referralThreshold

    #print("Stats Service Input")
    #print(input)    
    response = {}
    output_as_json={}
    try:
        response = requests.post(os.environ["STATS_ENDPOINT"], json=input)
    except requests.exceptions.RequestException as e: 
        print("ERROR: Stats Service Connection Failure")
        print(e)
        
    if(response.status_code==200 and response.text):
        output_as_txt=response.text[2:len(response.text)-2].replace("\'","\"") # remove square brackets and string qoutation at the begining and end
        output_as_json= json.loads(output_as_txt)
        #print("Stats Service Output")
        #print(output_as_json)
    else:
        print("ERROR: Stats Service Error")
        print("ERROR: Status Code - ", response.status_code)
        print("ERROR: output_txt - ", response.text)

    return output_as_json
    
def compute_rank_range(comparison_result):
    rank = []
    if comparison_result is None:
        return rank
    elif (np.count_nonzero(comparison_result) == len(comparison_result)):
        rank.append(len(comparison_result)+1) # min
        rank.append(len(comparison_result)+1) # max
    elif (np.count_nonzero(comparison_result) == 0):
        rank.append(0) #min
        rank.append(0) #max
    else:
        for i in range(len(comparison_result) - 1):
            if comparison_result[i] != comparison_result[i+1]:
                if i == 0:
                    rank.append(i + 1)
                else:
                    rank.append(i + 2)

    if len(rank) == 1: # for cases of 1 1 ... 1 0 0 ... 0
        rank.append(rank[0]) # min = max

    return rank
"""
def compute_rank_range(comparison_result):
    rank = []
    if comparison_result is None:
        return rank
    elif (np.count_nonzero(comparison_result) == len(comparison_result)):
        rank.append(0) #min
        rank.append(0) #max
    elif (np.count_nonzero(comparison_result) == 0):
        rank.append(len(comparison_result)+1) # min
        rank.append(len(comparison_result)+1) # max
    else:
        for i in range(len(comparison_result) - 1):
            if comparison_result[i] != comparison_result[i+1]:
                if i == 0:
                    rank.append(i + 1)
                else:
                    rank.append(i + 2)

    if len(rank) == 1: # for cases of 1 1 ... 1 0 0 ... 0
        rank.append(rank[0]) # min = max

    return rank
"""
def get_reference_data():

    response  = {}
    try:
         response  = api_service.call_sortdr_api(os.environ['IMAGES_ENDPOINT'])
    except requests.exceptions.RequestException as e: 
        print("ERROR: Image API Service Connection Failure")
        print(e)
        #os.exit(1)

    if response.status_code != 200:
        print("ERROR: Image API Service -  could not get test images data")
        print("ERROR: Image API response code - " + str(response.status_code))
        #os.exit(1)

    return response.json()

def verify_user(userToken):
    if userToken is None or request.remote_addr not in os.environ['ALLOWED_IP_ADDRESSES']:
        return False
    response  = {}
    try:
         response  = api_service.call_sortdr_api(os.environ['USERS_ENDPOINT'] + str(userToken))
    except requests.exceptions.RequestException as e: 
        print("ERROR: User API Service Connection Failure")
        print(e) 
    if response.status_code == 200 and response.content == b'true':
        return True
    else:
        print("ERROR: User API Service - could not verify user with the given userToken " + str(userToken))
        print("ERROR: User API response code - " + str(response.status_code))
        return False

