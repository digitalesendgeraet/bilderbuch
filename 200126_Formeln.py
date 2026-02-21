import numpy as np
from PIL import Image
import math
import random
import json
import os
import pandas as pd
import plotly.express as px         #you need to install module 'statsmodels' (python -m pip install statsmodels) for this to work

def sigmoid(vector):
    return 1 / (1+np.exp(-vector))
    
def derev_sigmoid(vector):
    s = sigmoid(vector)
    return s * (1-s)
    
def relu(vector):
    return np.where(vector > 0, vector, 0.01 * vector)

def derev_relu(vector):
    return np.where(vector > 0, 1, 0.01)


def write_json(data, filename):
    # allow passing string OR dict
    if isinstance(data, str):
        data = json.loads(data.replace("'", '"'))

    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data["time"].update(data)
        file.seek(0)
        json.dump(file_data, file, indent=4)
        file.truncate()

def showGraph():
    with open('learning.json', 'r') as file:
        data = json.load(file)

    time = data['time']

    df = pd.DataFrame.from_dict(time, orient='index')
    df = df.astype(float)

    fig = px.scatter(df,x=df.index, y=abs(df[0]))

    fig = px.scatter(
        df, x=df.index, y=abs(df[0]), opacity=0.65,
        trendline="ols", trendline_color_override='red', title='fehler'
    )

    fig.show()




class Layer:
    def __init__(self, size, prev_Size):
        self.bias = np.zeros((size, size))
        self.biases_sensitivity = np.zeros((size, size))
        self.values = np.zeros((size, size))
        self.value_sensitivity = np.zeros((size, size))

        self.weights = np.zeros((size, size, prev_Size, prev_Size))
        self.weights_sensitivity = np.zeros((size, size, prev_Size, prev_Size))

        self.z = np.zeros((size, size))
        self.goal = np.zeros((size, size))

        self.size = size
        self.prev_Size = prev_Size



    def next_layer(self, prev_Layer):
        self.z = np.einsum('ij,nmij->nm', prev_Layer.values, self.weights) + self.bias

        if self.size == 1:  # Output layer
            self.values = sigmoid(self.z)
        else:
            self.values = relu(self.z)


    
    def weight_sensitivity(self, prev_Layer): #prev_Layer ist layer davor (nichts umgedreht durch backpropagation)
        
        dz_nach_dw = prev_Layer.values

        if self.size == 1:
            da_nach_dz = derev_sigmoid(self.z)
            #da_nach_dz = 0.5
            dc_nach_da = 2 * (self.values-self.goal)
        else:
            da_nach_dz = derev_relu(self.z)
            dc_nach_da = self.value_sensitivity

        dc_nach_dz = np.multiply(dc_nach_da, da_nach_dz)

        self.weights_sensitivity = np.einsum('ij,nm->ijnm', dc_nach_dz, dz_nach_dw)



    
    def bias_sensitivity(self):
        dz_nach_db = 1

        if self.size == 1:
            da_nach_dz = derev_sigmoid(self.z)
            #da_nach_dz = 0.5
            dc_nach_da = 2 * (self.values-self.goal)
        else:
            da_nach_dz = derev_relu(self.z)
            dc_nach_da = self.value_sensitivity

        self.biases_sensitivity = dz_nach_db * np.multiply(dc_nach_da, da_nach_dz)



    def prev_Val_sensitivity(self, prev_Layer):
        dz_nach_da = self.weights

        if self.size == 1:
            da_nach_dz = derev_sigmoid(self.z)
            #da_nach_dz = 0.5
            dc_nach_da = 2 * (self.values-self.goal)
        else:
            da_nach_dz = derev_relu(self.z)
            dc_nach_da = self.value_sensitivity

        dc_nach_dz = np.multiply(dc_nach_da, da_nach_dz)

        prev_Layer.value_sensitivity = np.einsum('ij,ijnm->nm', dc_nach_dz, dz_nach_da)

        



class Network:

    def __init__(self):
        self.input_layer =  Layer(16, 1)
        self.hidden_layer = Layer(4, 16)
        self.output_layer = Layer(1, 4)                          # 0-index = True ; 1-index = Flase
        self.learning_rate = 0.001



    def generate_random(self):
        bias = [[round(random.uniform(-0.1,0.1), 3) for x in range(self.hidden_layer.size)] for x in range(self.hidden_layer.size)] 
        self.hidden_layer.bias = np.array(bias)  

        weights = [[[[round(random.uniform(-0.1,0.1), 3)  for x in range(self.hidden_layer.prev_Size)] for x in range(self.hidden_layer.prev_Size)] for x in range(self.hidden_layer.size)] for x in range(self.hidden_layer.size)]
        self.hidden_layer.weights = np.array(weights)

        bias = [[round(random.uniform(-0.1,0.1), 3)  for x in range(self.output_layer.size)] for x in range(self.output_layer.size)] 
        self.output_layer.bias = np.array(bias)  

        weights = [[[[round(random.uniform(-0.1,0.1), 3)  for x in range(self.output_layer.prev_Size)] for x in range(self.output_layer.prev_Size)] for x in range(self.output_layer.size)] for x in range(self.output_layer.size)]
        self.output_layer.weights = np.array(weights)  

        self.write_all()   



    def img_open(self, file):
        data_i = Image.open(file).convert("L")
        self.input_layer.values = np.array(data_i) / 255.0



    def read_all(self):
        self.output_layer.bias = np.load("output_bias.npy")
        self.output_layer.weights = np.load("output_weights.npy")

        self.hidden_layer.bias = np.load("hidden_bias.npy")
        self.hidden_layer.weights = np.load("hidden_weights.npy")




    def write_all(self):
        np.save("output_bias.npy", self.output_layer.bias)
        np.save("output_weights.npy", self.output_layer.weights)

        np.save("hidden_bias.npy", self.hidden_layer.bias)
        np.save("hidden_weights.npy", self.hidden_layer.weights)



    def run(self, file):
        #self.read_all()

        self.img_open("test_images/" + file)

        self.hidden_layer.next_layer(self.input_layer)
        self.output_layer.next_layer(self.hidden_layer)

        return self.output_layer.values
    
    


    def learning(self, file):
        self.hidden_layer.weights_sensitivity.fill(0)
        self.hidden_layer.biases_sensitivity.fill(0)
        self.hidden_layer.value_sensitivity.fill(0)

        self.output_layer.weights_sensitivity.fill(0)
        self.output_layer.biases_sensitivity.fill(0)
        self.output_layer.value_sensitivity.fill(0)


        with open('goals.json', 'r') as goals:
            data = json.load(goals)

        pictures = data['pictures']
        goal = pictures[file]["goal"]

        self.output_layer.goal = np.array([[goal]])

        self.run(file)

        self.output_layer.bias_sensitivity()
        self.output_layer.weight_sensitivity(self.hidden_layer)
        self.output_layer.prev_Val_sensitivity(self.hidden_layer)

        self.hidden_layer.bias_sensitivity()
        self.hidden_layer.weight_sensitivity(self.input_layer)

        self.hidden_layer.weights = self.hidden_layer.weights - self.learning_rate * self.hidden_layer.weights_sensitivity# + random.uniform(-0.00005, 0.00005)
        self.hidden_layer.bias = self.hidden_layer.bias - self.learning_rate * self.hidden_layer.biases_sensitivity# + random.uniform(-0.00005, 0.00005)

        self.output_layer.weights = self.output_layer.weights - self.learning_rate * self.output_layer.weights_sensitivity# + random.uniform(-0.00005, 0.00005)
        self.output_layer.bias = self.output_layer.bias - self.learning_rate * self.output_layer.biases_sensitivity# + random.uniform(-0.00005, 0.00005)

        

    

    def full_learning(self, epochen = 2000):
        self.read_all()

        file = "learning.json"

        data = {
            "time": {}
        }
        json_str = json.dumps(data, indent=4)
        with open(file, "w") as f:
            f.write(json_str)

        fehler = []

        for i in range(epochen):
            images = os.listdir("test_images")
            random.shuffle(images)
            for image in images:
                self.learning(image)

                fehler.append((self.output_layer.goal[0,0] - self.output_layer.values[0,0])**2)
            print(i)

        data = {}
        for j in range(len(fehler)):
            data.update({str(j): fehler[j]})
        write_json(data, "learning.json")

        self.write_all()

n = Network()
n.read_all()
for i in range(0,500, 2):
    ergebnis = n.run("test_"+str(i)+".png")
    if ergebnis <= 0.5:
        print(ergebnis)
        print(i)
#n.generate_random()
#n.full_learning()
#showGraph()
print("done")
