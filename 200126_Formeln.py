import numpy as np
import PIL as PL
import math

def sigmoid(x):
    return(1/(1+math.e**(-x)))


class Layer:

    def __init__(self):
        self.bias = np.array([None])                        # corosponds to b
        self.values = np.array([None])                      # corosponds to a
        self.weights = np.array([None])                     # corosponds to w
        self.z= np.array([None])                            # corosponds to z
        self.goal = np.array([None])                        # corosponds to y

class Network:

    def __init__(self):
        self.input_layer =  Layer()
        self.hidden_layer = Layer()
        self.output_layer = Layer()                          # 0-index = True ; 1-index = Flase

    def r_file(self,file):
        data_i = PL.Image.open(file)
        self.hidden_layer = np.array(data_i)

    def next_layer(self, prev_Layer, cur_Layer, weights):
        for n in range(len(cur_Layer)):
            for m in range(len(cur_Layer[0])):
                sum = 0
                for i in range(len(weights[n])):
                    for j in range(len(weights[n,0])):
                        sum += prev_Layer[i,j]*weights[n,i,j]
                sum += cur_Layer.bias[n, m]
                cur_Layer[n,m] = sigmoid(sum)

        return cur_Layer

    def run(self, file):
        self.r_file(file)
        self.hidden_layer = self.next_layer(self,self.input_layer, self.hidden_layer, self.edge_weight_i)
        self.output_layer = self.next_layer(self,self.hidden_layer, self.output_layer, self.edge_weight_o)
        trueVal = self.output_layer[0,0]
        falseVal = self.output_layer[1,0]
        return trueVal, falseVal
    
    def wieght_sensitivity(self):
