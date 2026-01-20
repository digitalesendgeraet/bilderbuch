import numpy as np
import PIL as PL
import math

def sigmoid(x):
    return(1/(1+math.e**(-x)))


class Layer:

    def __init__(self):
        self.bias = np.array([None])
        self.values = np.array([None])

class Network:

    def __init__(self):
        self.input_layer =  np.array([None])   
        self.hidden_layer = np.array([None])
        self.output_layer = np.array([[None],[None]]) # 0-index = True ; 1-index = Flase
        self.edge_weight_i = np.array([None])
        self.edge_weight_o = np.array([None])

    def r_file(self,file):
        data_i = PL.Image.open(file)
        self.hidden_layer = np.array(data_i)

    def next_layer(self, prev_Layer, cur_Layer):
        for n in range(len(cur_Layer.values)):
            for m in range(len(cur_Layer.values[n])):
                sum = 0
                for i in range(len(cur_Layer.weights[n])):
                    for j in range(len(cur_Layer.weights[n,m])):
                        sum += prev_Layer.values[i,j]*cur_Layer.weights[n,m,i,j]
                sum += cur_Layer.bias[n, m]
                cur_Layer.z[n,m] = sum
                cur_Layer.values[n,m] = sigmoid(sum)

        return cur_Layer

    def run(self, file):
        self.r_file(file)
        self.hidden_layer = self.next_layer(self,self.input_layer, self.hidden_layer, self.edge_weight_i)
        self.output_layer = self.next_layer(self,self.hidden_layer, self.output_layer, self.edge_weight_o)
        trueVal = self.output_layer[0,0]
        falseVal = self.output_layer[1,0]
        return trueVal, falseVal

    
        
