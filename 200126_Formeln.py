import numpy as np
import PIL as PL
import math

def sigmoid(x):
    return(1/(1+math.e**(-x)))

def derev_sigmoid(x):
    math.e**(-x)/(1+math.e**(-x)+math.e**(-2*x))



class Layer:
    def __init__(self, size, prev_Size):
        bias = [[None for x in range(size)] for x in range(size)] 
        self.bias = np.array(bias)                        # corosponds to b

        bias_sens = [[None for x in range(size)] for x in range(size)] 
        self.biases_sensitivity = np.array(bias_sens) 

        values = [[None for x in range(size)] for x in range(size)] 
        self.values = np.array(values)                      # corosponds to a

        values_sens = [[None for x in range(size)] for x in range(size)] 
        value_sensitivity = np.array(values_sens)

        weights = [[[[None for x in range(prev_Size)] for x in range(prev_Size)] for x in range(size)] for x in range(size)]
        self.weights = np.array(weights)                     # corosponds to w -> von vorheriger zu dieser layer

        weights_sens = [[[[None for x in range(prev_Size)] for x in range(prev_Size)] for x in range(size)] for x in range(size)]
        self.weights_sensitivity = np.array(weights_sens) 

        z = [[None for x in range(size)] for x in range(size)] 
        self.z= np.array(z)                            # corosponds to z

        goal = z = [[None for x in range(size)] for x in range(size)] 
        self.goal = np.array(goal)                        # corosponds to y

    def next_layer(self, prev_Layer):
        for n in range(len(self.values)):
            for m in range(len(self.values[n])):
                sum = 0
                for i in range(len(prev_Layer)):
                    for j in range(len(prev_Layer[i])):
                        sum += prev_Layer.values[i,j]*self.weights[n,m,i,j]
                sum += self.bias[n, m]
                self.z[n,m] = sum
                self.values[n,m] = sigmoid(sum)
    
    def weight_sensitivity(self, prev_Layer): #prev_Layer ist layer davor (nichts umgedreht durch backpropagation)
        for n in range(len(self.values)):
            for m in range(len(self.values[n])):
                for i in range(len(self.weights[n,m])):
                    for j in range(len(self.weights[n,m,i])):
                        dz_nach_dw = prev_Layer.values[n,m]
                        da_nach_dz = derev_sigmoid(self.z[n,m])
                        dc_nach_da = 2 * (self.values[n,m]-self.goal[n,m])
                        dc_nach_dw = dz_nach_dw * da_nach_dz * dc_nach_da
                        self.weights_sensitivity[n,m,i,j] =  dc_nach_dw
    
    def bias_sensitivity(self):
        for n in range(len(self.values)):
            for m in range(len(self.values[n])):
                dz_nach_db = 1
                da_nach_dz = derev_sigmoid(self.z[n,m])
                dc_nach_da = 2 * (self.values[n,m]-self.goal[n,m])
                dc_nach_db = dz_nach_db * da_nach_dz * dc_nach_da
                self.weights_sensitivity[n,m] =  dc_nach_db

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
    
    
