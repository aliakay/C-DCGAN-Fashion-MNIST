from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from Model import*
from generate_test import Generate_samples

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('generate_sample')

print("-----Welcome to Fashion Mnist data Generetor comment line-------- ")
print("Please open a new terminal,choose the category and write the example code in order to generate sample images")
print("-----CATEGORY-----> 'T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'")
print('EXAMPLE ---> curl -X GET "http://localhost:5000/?generate_sample=Sneaker" ' )

class GenerateSample(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['generate_sample']
        
        sample_images = Generate_samples(user_query)
        
        return print("image is generated and saved as"+user_query+"test.png")

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(GenerateSample, '/')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
