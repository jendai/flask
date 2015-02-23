import requests
import json

API_url = "http://104.131.124.105:5000/predict"
one_example = {"example": [192]}
payload = json.dumps(one_example)
headers = {'content-type': 'application/json'}

response = requests.post(API_url, data=payload, headers=headers)

# The status code: 200 is OK, 500 is internal server error,
# check others at http://en.wikipedia.org/wiki/List_of_HTTP_status_codes
print "The response status code is %s" % response.status_code

# The returned content of the response
print "The full response from the API is:"
print response.text

# convert this to a dict, get the info you need out, print it
output = response.json()
predicted_class = output["predicted"][0]
print "The predicted class for x=192 is %i!" % predicted_class


# Let's do this again for another example:
new_x = 8916
payload = json.dumps( {"example": [new_x]} )
response = requests.post(API_url, data=payload, headers=headers)
predicted_class = response.json()["predicted"][0]
print "The predicted class for x=%i is %i!" % (new_x, predicted_class)

