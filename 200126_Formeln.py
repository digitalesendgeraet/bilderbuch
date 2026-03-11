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
    
def relu(vector): # Leaky Relu, damit wert nie Null wird damit keine "toten" Neuronen entstehen
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

    fig = px.scatter(df,x=df.index, y=abs(df[0])) #Trendlinie

    fig = px.scatter(
        df, x=df.index, y=abs(df[0]), opacity=0.65,
        trendline="ols", trendline_color_override='red', title='fehler'
    )

    fig.show()




class Layer:
    def __init__(self, size, prev_Size): #Netzstrukutur: Jede Layer 2D (weil InputLayer (Bild) 2D sein sollte) 
        self.bias = np.zeros((size, size)) #Bias nicht in Weights eingeschlossen, da unsere Informationsquelle dies nicht als möglichkeit genannt hat
        self.biases_sensitivity = np.zeros((size, size)) #Sensitvity Werte: basicly Korrektur, wie sehr verändert eine änderung dieses Wertes das Ergebnis (-> Möglichst günstige Änderung)
        self.values = np.zeros((size, size))
        self.value_sensitivity = np.zeros((size, size))

        self.weights = np.zeros((size, size, prev_Size, prev_Size)) #Da jede Layer 2D ist die Verbindung zwischen den Layer 4D, Struktur, erst jetzige Layer und darin vorherige Layer
        self.weights_sensitivity = np.zeros((size, size, prev_Size, prev_Size))

        self.z = np.zeros((size, size)) #beim Durchlauf values ohne Sigmoid bzw. Railu -> für Lernen sinnvoll, da es gebraucht wird und nicht neu berechnet werden muss
        self.goal = np.zeros((size, size))

        self.size = size
        self.prev_Size = prev_Size



    def next_layer(self, prev_Layer):
        self.z = np.einsum('ij,nmij->nm', prev_Layer.values, self.weights) + self.bias #Rechnung um von einer Layer zur nächsten zu kommen

        if self.size == 1:  # Output layer: Sigmoid statt Railu, damit Fehler besser zwischen 0 und 1 liegt wodurch lerenen und Fehlerberechen besser funktioniert
            self.values = sigmoid(self.z)
        else: # Alle anderen Layers: Relu, da es effizienter ist und schneller lernt
            self.values = relu(self.z)


    
    def weight_sensitivity(self, prev_Layer): #prev_Layer ist layer davor (nichts umgedreht durch backpropagation)
        #Ableitung in Teile aufgeteilt, damit es übersichtlicher ist
        dz_nach_dw = prev_Layer.values

        if self.size == 1: # Outputlayer
            da_nach_dz = derev_sigmoid(self.z)
            #da_nach_dz = 0.5                                # -> zwischenduch versuch, war nicht besser
            dc_nach_da = 2 * (self.values-self.goal)        # Fehler
        else: # vorherige Layers
            da_nach_dz = derev_relu(self.z)
            dc_nach_da = self.value_sensitivity             # statt Fehler wird genommen wie sehr eine änderung der aktivierungen den Fehler beeinfulssen würde, da man nicht in outputLayer sondern davor ist

        dc_nach_dz = np.multiply(dc_nach_da, da_nach_dz) # Zusammenrechenen der Ableitungen

        self.weights_sensitivity = np.einsum('ij,nm->ijnm', dc_nach_dz, dz_nach_dw)    #dc_nach_dz und dz_nach_dw beschreiben verschiedene Formen für die Weights (da es 4D) (dc_nach_dz ist diese Layer; dz_nach_dw ist prevLayer), deshalb Einsum



    
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



    def prev_Val_sensitivity(self, prev_Layer):    #Berechenen wie sehr die Vorherigen Gewichte den Fehler beeinflussen würden, wichtig fürs Lernen in den vorherigen Layern
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
        self.input_layer =  Layer(100, 1)                        #100 x 100 Bild, (1 nur als Füllerwert)
        self.hidden_layer = Layer(8, 100)                        # eine Hiddenlayer 8 x 8
        self.output_layer = Layer(1, 8)                          # 1x1 Outputlayer: 1 -> reflektirende Kugel; 0 -> keine reflektirende Kugel
        self.learning_rate = 0.001                               # sehr geringe Lernrate, wegen langem lernen



    def generate_random(self): #Generieren von Random Startgewichten und Biasen zum Anfang vom Lernen; Startwerte nah an 0
        bias = [[round(random.uniform(-0.1,0.1), 3) for x in range(self.hidden_layer.size)] for x in range(self.hidden_layer.size)] 
        self.hidden_layer.bias = np.array(bias)  

        weights = [[[[round(random.uniform(-0.1, 0.1), 3)  for x in range(self.hidden_layer.prev_Size)] for x in range(self.hidden_layer.prev_Size)] for x in range(self.hidden_layer.size)] for x in range(self.hidden_layer.size)]
        self.hidden_layer.weights = np.array(weights)

        bias = [[round(random.uniform(-0.1,0.1), 3)  for x in range(self.output_layer.size)] for x in range(self.output_layer.size)] 
        self.output_layer.bias = np.array(bias)  

        weights = [[[[round(random.uniform(-0.1,0.1), 3)  for x in range(self.output_layer.prev_Size)] for x in range(self.output_layer.prev_Size)] for x in range(self.output_layer.size)] for x in range(self.output_layer.size)]
        self.output_layer.weights = np.array(weights)  

        self.write_all()   



    def img_open(self, file):
        data_i = Image.open(file)
        self.input_layer.values = np.array(data_i) / 255.0



    def read_all(self): # Weights und Biases in npy Datein gespichert, damit gleiche Wieghts genutzt werden können ohne Programm immer neu laufen zu Lassen
        self.output_layer.bias = np.load("output_bias.npy")
        self.output_layer.weights = np.load("output_weights.npy")

        self.hidden_layer.bias = np.load("hidden_bias.npy")
        self.hidden_layer.weights = np.load("hidden_weights.npy")




    def write_all(self): # npy Datein sind ein Numpy Format welches dafür da ist arrays zu speichern 
        np.save("output_bias.npy", self.output_layer.bias)
        np.save("output_weights.npy", self.output_layer.weights)

        np.save("hidden_bias.npy", self.hidden_layer.bias)
        np.save("hidden_weights.npy", self.hidden_layer.weights)



    def run(self, file): #Einmal Bild vorhersagen
        #self.read_all()

        self.img_open("formated_images/" + file)

        #Durch Layers durchgehen
        self.hidden_layer.next_layer(self.input_layer)
        self.output_layer.next_layer(self.hidden_layer)

        return self.output_layer.values
    
    


    def learning(self, file):
        # Immer Sensitivityes auf Null setzen, da diese nur für 1 anpassen sinnvoll sind
        self.hidden_layer.weights_sensitivity.fill(0)
        self.hidden_layer.biases_sensitivity.fill(0)
        self.hidden_layer.value_sensitivity.fill(0)

        self.output_layer.weights_sensitivity.fill(0)
        self.output_layer.biases_sensitivity.fill(0)
        self.output_layer.value_sensitivity.fill(0)

        #Einlesen vom Goal
        with open('goals.json', 'r') as goals:
            data = json.load(goals)

        pictures = data['pictures']
        goal = pictures[file]["goal"]

        self.output_layer.goal = np.array([[goal]])
        
        self.run(file)    #Durchlaufen

        #Berechenen wie sehr die Veränderung der Werte eine Veränderung des Fehlers erziehlt
        self.output_layer.bias_sensitivity()
        self.output_layer.weight_sensitivity(self.hidden_layer)
        self.output_layer.prev_Val_sensitivity(self.hidden_layer)

        self.hidden_layer.bias_sensitivity()
        self.hidden_layer.weight_sensitivity(self.input_layer)

        #Anpassen der weights und Biases mit dem Berechneten; hatten mit zusätzlichem Random versucht, war nicht besser
        self.hidden_layer.weights = self.hidden_layer.weights - self.learning_rate * self.hidden_layer.weights_sensitivity# + random.uniform(-0.00005, 0.00005)
        self.hidden_layer.bias = self.hidden_layer.bias - self.learning_rate * self.hidden_layer.biases_sensitivity# + random.uniform(-0.00005, 0.00005)

        self.output_layer.weights = self.output_layer.weights - self.learning_rate * self.output_layer.weights_sensitivity# + random.uniform(-0.00005, 0.00005)
        self.output_layer.bias = self.output_layer.bias - self.learning_rate * self.output_layer.biases_sensitivity# + random.uniform(-0.00005, 0.00005)

        

    

    def full_learning(self, epochen = 3000):
        self.read_all() #Einlesen der Weights und Biases

        #Lernverlauf in Json Datei zwischenspeichern: Json Erstellen
        file = "learning.json"

        data = {
            "time": {}
        }
        json_str = json.dumps(data, indent=4)
        with open(file, "w") as f:
            f.write(json_str)

        fehler = []    #Fehlerverlauf speichern

        for i in range(epochen):
            print(i)
            images = os.listdir("formated_images")
            random.shuffle(images)                #In jeder Epoche alle unsere Bilder einmal durchgehen, und shuffel, damit wir nicht zyklen erzeugen die das Lernen verschlechtern
            for image in images:
                self.learning(image)    #Lernen

                fehler.append((self.output_layer.goal[0,0] - self.output_layer.values[0,0])**2)    #Fehler für die Json

        #Am Ende einmal Fehlerverlauf in Json speichern für öfteres ansehen
        data = {}
        for j in range(len(fehler)):
            data.update({str(j): fehler[j]})
        write_json(data, "learning.json")

        self.write_all()    #Berechnete Gewichte Speichern, damit diese als Trainierte Gewichte genutzt werden können


n = Network()
#n.generate_random()       
n.read_all()
#n.full_learning()
#print(n.input_layer.bias)

#Alle Falsch sortierten Bilder oder unsicher Sortierten nach dem Gelernt wurde bestimmen (um Anzahl zu sehen)
images = os.listdir("formated_images")
for image in images:
    output = n.run(image)

    with open('goals.json', 'r') as goals:
            data = json.load(goals)

    pictures = data['pictures']
    goal = pictures[image]["goal"]

    if (goal - output[0,0])**2 > 0.1:
        print(image)
        print((goal - output[0,0])**2)


#print(round(n.run("test_480.png")[0,0], 5))        
n.read_all()
#showGraph()

print("done")
