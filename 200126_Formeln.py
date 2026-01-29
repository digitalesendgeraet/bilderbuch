import numpy as np
import PIL as PL
import math

def sigmoid(x):
    return (1/(1+math.e**(-x)))

def derev_sigmoid(x):
    return (- math.e**(-x)/(1+math.e**(-x)+math.e**(-2*x)))



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
    
    def weight_sensitivity(self, prev_Layer): #prev_Layer ist layer davor (nichts umgedreht durch backpropagation)
        for n in range(len(prev_Layer.values)):
            for m in range(len(prev_Layer.values[n])):
                for i in range(len(self.values)):
                    for j in range(len(self.values[i])):
                        dz_nach_dw = prev_Layer.values[n,m] 
                        da_nach_dz = derev_sigmoid(self.z[i, j])
                        dc_nach_da = 2 * (self.values[i, j]-self.goal[i, j])
                        dc_nach_dw = dz_nach_dw * da_nach_dz * dc_nach_da
                        self.weights_sensitivity[i, j, n, m] =  dc_nach_dw
    
    def bias_sensitivity(self):
        for n in range(len(self.values)):
            for m in range(len(self.values[n])):
                dz_nach_db = 1
                da_nach_dz = derev_sigmoid(self.z[n,m])
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
        self.input_layer =  Layer(3, 1)
        self.hidden_layer = Layer(2, 3)
        self.output_layer = Layer(1, 2)                          # 0-index = True ; 1-index = Flase
        self.learning_rate = 0.5

    def img_open(self, file):
        data_i = PL.Image.open(file)
        self.input_layer.values = np.array(data_i)

    def r_file(self, file, layer):
        layer = np.array(data_i)

    def run(self):
        #self.img_open(file)
        #self.r_file(file, self.output_layer.weights)
        #self.r_file(file, self.output_layer.bias)



        
        

        self.hidden_layer.next_layer(self.input_layer)
        self.output_layer.next_layer(self.hidden_layer)



        trueVal = self.output_layer.values
        return trueVal
    

    def learning(self):
        self.output_layer.bias = np.array([[-8]])
        self.output_layer.weights = np.array([[[[0,1], [0,2]]]])

        self.hidden_layer.bias = np.array([[-8, 2], [-4, 6]])
        self.hidden_layer.weights = np.array([[[[0,1,5], [0,2,1], [2,0,1]], [[0,1,5], [0,2,1], [2,0,1]]], [[[0,1,5], [0,2,1], [2,0,1]], [[0,1,5], [0,2,1], [2,0,1]]]])

        self.input_layer.values = np.array([[0,0,0], [1,1,1], [1,1,1]])

        self.output_layer.goal = np.array([[0]])



        self.run()

        self.output_layer.bias_sensitivity()

        self.output_layer.weight_sensitivity(self.hidden_layer)

        self.output_layer.prev_Val_sensitivity(self.hidden_layer)

        print(self.output_layer.biases_sensitivity)
        print(self.output_layer.weights_sensitivity)
        print(self.hidden_layer.value_sensitivity)

    

n = Network()
n.learning()
    
