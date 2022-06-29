from __future__ import print_function
import sys
import base64
import datetime
import json as json2
import flask
from flask import request, json

import comparison_service as service

print("getting all reference data")
start_time = datetime.datetime.now()
reference_data=service.get_reference_data()

if not reference_data:
    sys.exit(0)

end_time = datetime.datetime.now()
time_taken = end_time-start_time
print("time taken for all reference data: ", str(time_taken))
print("size of the reference data: "+str(len(reference_data)))

import pandas as pd
import numpy as np
import os
import math


reflist = ["A", "B", "C", "D", "E", "F", "G"]
mydf_lookup = pd.read_excel ("23_pro.xlsx")
rawavgs = pd.read_excel ("rawavgs.xlsx")

n = len(mydf_lookup[mydf_lookup["left_label"] <= 2])

GNGO = pd.read_excel ("GNGO_23.xlsx")
GNGO_max = GNGO["GNGO_Max"].values[0]
GNGO_min = GNGO["GNGO_Min"].values[0]

logprob_0123 = np.log (len (mydf_lookup[:n])/len(mydf_lookup))
logprob_34 = np.log (len(mydf_lookup[n+1:])/len(mydf_lookup))
print (logprob_0123)
print (logprob_34)

def getresult2(rawdata_str, lookup):
    # THIS IS THE LOOKUP TABLE
    resultarray = [[],[],[],[]]
    numericalarray = [[],[],[],[]]
    predictedvaluearray = []
    colname = []
    idx  = 0
    # print (rawdata_str)
    rawdata = rawdata_str["ComparisonResult"]

    # mydf_oneline = pd.read_excel ("./sample_data/TRAINING 23_PRO.xlsx")
    # THIS IS THE DATA TABLE
#     mydf_oneline = pd.read_excel ("./drive/MyDrive/6_pro.xlsx")
    for t in rawdata:
        residx = 0
        residx1 = 0
        predictedvaluearray.append (t["predictedValue"])
        colname.append (str(t["weightedScore"]) + "." + str(idx))
        for val in t["multipleProbabilityValues"]:
            resultarray[residx].append (1-val)
            residx = residx + 1
        for val1 in t["multiplePredictedValues"]:
            numericalarray[residx1].append (1-val1)
            residx1 = residx1 + 1
        idx  = idx + 1


    mydf = pd.DataFrame ( resultarray, columns=colname)

    mydf1 = pd.DataFrame (numericalarray)

    pstr = ""
    rawstr = mydf.to_json()
    
    for pval in predictedvaluearray:
        pstr = pstr + str(pval) + " , " 
    mydf.to_excel ("rawresults.xlsx", index = False)
    
    if lookup == False:
        mydf_oneline = mydf
    else:
        mydf_oneline = mydf_lookup

    # mydf_full = mydf_lookup [["A", "B", "C", "D", "E", "F", "G", "left_label"]]
    avglist = [[],[],[],[],[],[],[]]
    avglist_data = [[],[],[],[],[],[],[]]
    print ("oneline: " + str(mydf_oneline))
    for mycol in mydf_oneline.columns:
      colname  = str(mycol) + "."
      try:
        colnum = int (str(colname[:colname.find(".")]))
        if colnum == 0:
          avglist_data[0].append ((mycol))
        elif colnum >=1 and colnum < 14:
          avglist_data[1].append ((mycol))
        elif colnum >= 14 and colnum <=49:
          avglist_data[2].append ((mycol))
        elif colnum >= 52 and colnum <=103:
          avglist_data[3].append ((mycol))
        elif colnum >= 127 and colnum <= 476:
          avglist_data[4].append ((mycol))
        elif colnum >= 516 and colnum <= 2944:
          avglist_data[5].append ((mycol))
        elif colnum == 3600:
          avglist_data[6].append ((mycol))
      except:
        print ("err")
        u=0


    flist_data = [[],[],[],[],[],[],[]]
    std_data = [[],[],[],[],[],[],[]]
    for idx,drow in mydf_oneline.iterrows():
      for f in range(0,7):
        numrow = drow[avglist_data[f]]
        avg = np.mean (numrow)
        n_std = np.std (numrow)
        flist_data[f].append (avg)
        std_data[f].append (n_std)
    mydf_full = pd.DataFrame()
    mydf_oneline = pd.DataFrame()
    for t in range(0,7):
      mydf_oneline[reflist[t]] = flist_data[t]

    rowlist = []

    likelihoodlist = [[],[],[],[],[],[],[],[]]
    loglikelihoodlist = [[],[],[],[],[],[],[],[]]

    likelihoodlist_a = [[],[],[],[],[],[],[],[]]
    loglikelihoodlist_a = [[],[],[],[],[],[],[],[]]
    avglist = [[],[],[],[],[],[],[]]
    stdlist = [[],[],[],[],[],[],[]]
    avgdf = pd.DataFrame(columns=["std_012", "std_34", "avg_012", "avg_34"])
    
    for f in range (0,7):

      standard_012 = rawavgs["std_012"].to_list()[f]
      standard_34 = rawavgs["std_34"].to_list()[f]
      myavg_012 = rawavgs["avg_012"].to_list()[f]
      myavg_34 = rawavgs["avg_34"].to_list()[f]


      print ("myavg_012 = " + str(myavg_012))
      print ("standard_012 = " + str(standard_012))  
      print ("myavg_34 = " + str(myavg_34))
      print ("standard_34 = " + str(standard_34))  
      # print ("average 34 = " + str(myavg_34))
      # print ("standard_34=" + str(standard_34))
#       standard_012 = rawavgs["std_012"].to_list()[f]
#       standard_34 = rawavgs["std_34"].to_list()[f]
#       myavg_012 = rawavgs["avg_012"].to_list()[f]
#       myavg_34 = rawavgs["avg_34"].to_list()[f]

      likelihood_12 = 1/np.sqrt(2*np.pi * (standard_012**2))
      likelihood_34 = 1/np.sqrt(2*np.pi * (standard_34**2))
      print ("likelihood_12 = " + str(likelihood_12 ))
      print ("likelihood_34 = " + str(likelihood_34 ))
      for lineavg in range (len(mydf_oneline)):
        exponent = np.e **-((mydf_oneline[reflist[f]].iloc[lineavg]-myavg_012)**2/(2*standard_012**2))
        likelihoodlist[f].append (likelihood_12 * exponent)
        loglikelihoodlist[f].append (np.log(likelihood_12 * exponent))

        exponent_2 = np.e **-((mydf_oneline[reflist[f]].iloc[lineavg]-myavg_34)**2/(2*standard_34**2))
        likelihoodlist_a[f].append (likelihood_34 * exponent_2)
        loglikelihoodlist_a[f].append (np.log(likelihood_34 * exponent_2))
 
    mydf_fin = pd.DataFrame ()
    for x in range(0,7):
      # mydf["012-likelihood-" + str(reflist[x])] = likelihoodlist[x]
      mydf_fin["avg_"+str(reflist[x])] = flist_data[x]
      mydf_fin["std_"+str(reflist[x])] = std_data[x]
      mydf_fin["012-loglikelihood-" + str(reflist[x])]= loglikelihoodlist[x]
      # mydf["34-likelihood-" + str(reflist[x])] = likelihoodlist_a[x]
      mydf_fin["34-loglikelihood-" + str(reflist[x])]= loglikelihoodlist_a[x]
    sum_12_list = []
    sum_34_list = []
    # mydf_fin["label"] = labellist
    difflist = []
    reslist = []


    for idx, row in mydf_fin.iterrows():
      sum_012 = 0
      sum_34 = 0
      for n in ["A","B", "C", "D", "E", "F", "G"]:
        sum_012 = sum_012 + row["012-loglikelihood-" + str(n)]
        sum_34 = sum_34 + row ["34-loglikelihood-" + str(n)]
      GN = sum_012 + logprob_0123
      GO = sum_34 + logprob_34
      result = GN-GO


      # if ((result<0) and ((row["label"] >=0) and (row["label"] < 3)) or (result > 0) and (row["label"] >2)):
      #   reslist.append (0)
      # elif ((result < 0) and (row["label"] >2) or (result>0) and ((row["label"] >=0) and (row["label"] < 3))):
      #   reslist.append(1)

      sum_12_list.append (GN)
      sum_34_list.append (GO)
      difflist.append (GN-GO)
    mydf_fin["Sum_012"] = sum_12_list
    mydf_fin ["Sum_34"] = sum_34_list
    mydf_fin ["GN-GO"] = difflist
    if (lookup == True):
        mydf_fin ["left_label"] = mydf_lookup["left_label"].to_list()
    resstr = ""
    nummodels = 0
    modeltotal = 0
    print ("** Difflist: " + str(difflist))
    retstr = str(np.mean(difflist))
    print ("** return string ** " + retstr)
    return (resstr)

class PredictionService(object):
    @classmethod
    def get_reference_data(self):
        return reference_data

    @classmethod
    def get_input(self,image_data):
        return base64.b64encode(image_data)

    @classmethod
    def get_output(self,predictions,referralThreshold,reference_data):
        return service.get_output(predictions,referralThreshold,reference_data)

    @classmethod
    def predict(self, image_data,referralThreshold):
        """For the input, do the predictions and return them."""
        image=self.get_input(image_data)

        reference_data= self.get_reference_data()
        if reference_data is None or len(reference_data)==0:
            print("No reference data exists")
        else:
            predictions = service.get_predictions(image)
            return self.get_output(predictions,referralThreshold,reference_data)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/rank', methods=['POST'])
def predict():
    userToken = request.headers['user']
    authorised = service.verify_user(userToken)
    if not authorised:
        err_msg="unauthorised request, made from IP: " + request.remote_addr + " with invalid user token: " + userToken
        print(err_msg)
        return flask.Response(response=err_msg, status=405, mimetype='application/json')

    image_data = request.data
    if image_data is None or len(image_data) == 0 or request.content_type != 'image/jpeg':
        return flask.Response(response='Unsupported file. The file must be an image of type jpg', status=500, mimetype='application/json')

    referralThreshold = request.headers['referralThreshold']
    if not referralThreshold:
        referralThreshold = "Default"

    output = PredictionService.predict(image_data,referralThreshold)
    if(output):
        # json.dumps (output["ComparisonResult"])

        compareresult = getresult2 (output, False)
        output["LogLikelihood"] = compareresult
        output["GNGO_Max"] = str(GNGO_max)
        output["GNGO_Min"] = str(GNGO_min)
        return (output)
    else:
        return flask.Response(response='ERROR: Unable to process your request', status=500, mimetype='application/json')


@app.route('/process2', methods=['GET'])
def process2():
    argument = request.args.get('rawdata')
    compareresult = os.getcwd()
    # rawavgs = pd.DataFrame ([1,2,3],[1,2,3])
    output = os.listdir()
    
    return (output)



@app.route('/status', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = PredictionService.get_reference_data() is not None

    status = 200 if health else 404
    return flask.Response(response='connected! loaded images: ' +str(len(PredictionService.get_reference_data())), status=status, mimetype='application/json')
