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

    def next_layer(self, prev_Layer):
        for n in range(len(self.values)):
            for m in range(len(self.values[n])):
                sum = 0
                for i in range(len(self.weights[n])):
                    for j in range(len(self.weights[n,m])):
                        sum += prev_Layer.values[i,j]*self.weights[n,m,i,j]
                sum += self.bias[n, m]
                self.z[n,m] = sum
                self.values[n,m] = sigmoid(sum)



class Network:

    def __init__(self):
        self.input_layer =  Layer()
        self.hidden_layer = Layer()
        self.output_layer = Layer()                          # 0-index = True ; 1-index = Flase

    def r_file(self,file):
        data_i = PL.Image.open(file)
        self.hidden_layer = np.array(data_i)

    def run(self, file):
        self.r_file(file)
        self.hidden_layer = self.next_layer(self,self.input_layer, self.hidden_layer, self.edge_weight_i)
        self.output_layer = self.next_layer(self,self.hidden_layer, self.output_layer, self.edge_weight_o)
        trueVal = self.output_layer[0,0]
        falseVal = self.output_layer[1,0]
        return trueVal, falseVal
    
    def wieght_sensitivity(self):
