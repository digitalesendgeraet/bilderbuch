import numpy as np
from PIL import Image
import math
import random
import json
import os

def sigmoid(x):
    return (1/(1+math.e**(-x)))

def derev_sigmoid(x):
    return (- math.e**(-x)/(1+math.e**(-x)+math.e**(-2*x)))

def r_file(file):
    with open(file, "r") as f:
        data = f.read()
    return np.array(data)
    
def w_file(file, data):
    with open(file, "w") as f:
        f.write(data)



class Layer:
    def __init__(self, size, prev_Size):
        bias = [[None for x in range(size)] for x in range(size)] 
        self.bias = np.array(bias)                        # corosponds to b

        bias_sens = [[None for x in range(size)] for x in range(size)] 
        self.biases_sensitivity = np.array(bias_sens) 

        values = [[None for x in range(size)] for x in range(size)] 
        self.values = np.array(values)                      # corosponds to a

        values_sens = [[None for x in range(size)] for x in range(size)] 
        self.value_sensitivity = np.array(values_sens)

        weights = [[[[None for x in range(prev_Size)] for x in range(prev_Size)] for x in range(size)] for x in range(size)]
        self.weights = np.array(weights)                     # corosponds to w -> von vorheriger zu dieser layer

        weights_sens = [[[[None for x in range(prev_Size)] for x in range(prev_Size)] for x in range(size)] for x in range(size)]
        self.weights_sensitivity = np.array(weights_sens) 

        z = [[None for x in range(size)] for x in range(size)] 
        self.z= np.array(z)                            # corosponds to z

        goal = z = [[None for x in range(size)] for x in range(size)] 
        self.goal = np.array(goal)                        # corosponds to y

        self.size = size
        self.prev_Size = prev_Size

    def next_layer(self, prev_Layer):
        for n in range(len(self.values)):
            for m in range(len(self.values[n])):
                sum = 0
                for i in range(len(prev_Layer.values)):
                    for j in range(len(prev_Layer.values[i])):
                        sum += prev_Layer.values[i,j]*self.weights[n,m,i,j]
                sum += self.bias[n, m]
                self.z[n,m] = sum
                self.values[n,m] = sigmoid(sum)
    
    def weight_sensitivity(self, prev_Layer, use_value_sensitivity=False): #prev_Layer ist layer davor (nichts umgedreht durch backpropagation)
        for n in range(len(prev_Layer.values)):
            for m in range(len(prev_Layer.values[n])):
                for i in range(len(self.values)):
                    for j in range(len(self.values[i])):
                        dz_nach_dw = prev_Layer.values[n,m] 
                        da_nach_dz = derev_sigmoid(self.z[i, j])
                        if use_value_sensitivity:
                            dc_nach_da = self.value_sensitivity[i,j]
                        else:
                            dc_nach_da = 2 * (self.values[i, j]-self.goal[i, j])
                        dc_nach_dw = dz_nach_dw * da_nach_dz * dc_nach_da
                        self.weights_sensitivity[i, j, n, m] =  dc_nach_dw
    
    def bias_sensitivity(self, use_value_sensitivity=False):
        for n in range(len(self.values)):
            for m in range(len(self.values[n])):
                dz_nach_db = 1
                da_nach_dz = derev_sigmoid(self.z[n,m])
                if use_value_sensitivity:
                    dc_nach_da = self.value_sensitivity[n,m]
                else:
                    dc_nach_da = 2 * (self.values[n,m]-self.goal[n,m])
                dc_nach_db = dz_nach_db * da_nach_dz * dc_nach_da
                self.biases_sensitivity[n,m] =  dc_nach_db

    def prev_Val_sensitivity(self, prev_Layer):
        for n in range(len(prev_Layer.values)):
            for m in range(len(prev_Layer.values[n])):
                summe = 0
                for i in range(len(self.values)):
                    for j in range(len(self.values[i])):
                        dz_nach_da = self.weights[i,j,n,m]
                        da_nach_dz = derev_sigmoid(self.z[i,j])
                        dc_nach_da = 2 * (self.values[i,j]-self.goal[i,j])
                        summe += dz_nach_da * da_nach_dz * dc_nach_da
                dc_nach_dw = summe
                prev_Layer.value_sensitivity[n,m] =  dc_nach_dw

        



class Network:

    def __init__(self):
        self.input_layer =  Layer(16, 1)
        self.hidden_layer = Layer(10, 16)
        self.output_layer = Layer(1, 10)                          # 0-index = True ; 1-index = Flase
        self.learning_rate = 0.5



    def generate_random(self):
        bias = [[round(random.uniform(-1,1), 3) for x in range(self.hidden_layer.size)] for x in range(self.hidden_layer.size)] 
        self.hidden_layer.bias = np.array(bias)  

        weights = [[[[round(random.uniform(-1,1), 3)  for x in range(self.hidden_layer.prev_Size)] for x in range(self.hidden_layer.prev_Size)] for x in range(self.hidden_layer.size)] for x in range(self.hidden_layer.size)]
        self.hidden_layer.weights = np.array(weights)

        bias = [[round(random.uniform(-1,1), 3)  for x in range(self.output_layer.size)] for x in range(self.output_layer.size)] 
        self.output_layer.bias = np.array(bias)  

        weights = [[[[round(random.uniform(-1,1), 3)  for x in range(self.output_layer.prev_Size)] for x in range(self.output_layer.prev_Size)] for x in range(self.output_layer.size)] for x in range(self.output_layer.size)]
        self.output_layer.weights = np.array(weights)  

        self.write_all()   



    def img_open(self, file):
        data_i = Image.open(file).convert("L")
        self.input_layer.values = np.array(data_i)



    def read_all(self):
        self.output_layer.bias = np.array(r_file("output_bias.txt"))
        self.output_layer.weights = np.array(r_file("output_weights.txt"))

        self.hidden_layer.bias = np.array(r_file("hidden_bias.txt"))
        self.hidden_layer.weights = np.array(r_file("hidden_weights.txt"))



    def write_all(self):
        w_file("output_bias.txt", str(self.output_layer.bias))
        w_file("output_weights.txt", str(self.output_layer.weights))

        w_file("hidden_bias.txt", str(self.hidden_layer.bias))
        w_file("hidden_weights.txt", str(self.hidden_layer.weights))



    def run(self, file):

        self.img_open("images/" + file)

        self.hidden_layer.next_layer(self.input_layer)
        self.output_layer.next_layer(self.hidden_layer)

        trueVal = self.output_layer.values
        return trueVal
    
    


    def learning(self, file):

        with open('goals.json', 'r') as file:
            data = json.load(file)

        pictures = data['pictures']
        goal = pictures[file]["goal"]

        self.output_layer.goal = np.array([[goal]])

        self.run(file)

        self.output_layer.bias_sensitivity()
        self.output_layer.weight_sensitivity(self.hidden_layer)
        self.output_layer.prev_Val_sensitivity(self.hidden_layer)

        self.hidden_layer.bias_sensitivity(use_value_sensitivity=True)
        self.hidden_layer.weight_sensitivity(self.input_layer, use_value_sensitivity=True)

        self.hidden_layer.weights = self.hidden_layer.weights - self.learning_rate * self.hidden_layer.weights_sensitivity
        self.hidden_layer.bias = self.hidden_layer.bias - self.learning_rate * self.hidden_layer.biases_sensitivity

        self.output_layer.weights = self.output_layer.weights - self.learning_rate * self.output_layer.weights_sensitivity
        self.output_layer.bias = self.output_layer.bias - self.learning_rate * self.output_layer.biases_sensitivity

        self.write_all()

    

    def full_learning(self):
        for image in os.listdir("images"):
            self.learning(image)

n = Network()

# file = "goals.json"           #das muss in die datei von nelly rein, die die bilder erstellt (automatisches abspeichern des goals in einer json datei)

# print("start")
# data = {
#         "pictures": 
#         {}
#     }

# json_str = json.dumps(data, indent=4)
# with open(file, "w") as f:
#     f.write(json_str)

# file_name = "test_0.png"
# goal = 1
# data = {file_name:{"goal": goal}}

# def write_json(data, filename="goals.json"):
#     with open(filename, 'r+') as file:
#         file_data = json.load(file)
#         file_data["pictures"].update(data)
#         file.seek(0)
#         json.dump(file_data, file, indent=4)
