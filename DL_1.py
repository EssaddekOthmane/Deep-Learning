import pandas as pd
import streamlit as st 
import altair as alt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from PIL import Image



st.title("Constructeur de modèle profond de prédiction")
st.subheader("1. Introduction")
st.markdown("Le deep learning ou apprentissage profond est un sous-domaine de l'intelligence artificielle (IA). Ce terme désigne l'ensemble des techniques d'apprentissage automatique (machine learning), autrement dit une forme d'apprentissage fondée sur des approches mathématiques, utilisées pour modéliser des données. Pour mieux comprendre ces techniques, il faut remonter aux origines de l'intelligence artificielle en 1950, année pendant laquelle Alan Turning s'intéresse aux machines capables de penser")
st.markdown("Cette réflexion va donner naissance au machine learning, une machine qui communique et se comporte en fonction des informations stockées. Le deep learning est un système avancé basé sur le cerveau humain, qui comporte un vaste réseau de neurones artificiels. Ces neurones sont interconnectés pour traiter et mémoriser des informations, comparer des problèmes ou situations quelconques avec des situations similaires passées, analyser les solutions et résoudre le problème de la meilleure façon possible.")
st.subheader("2. Application à une base de donnéé avec cibles multiple (2)")
df=pd.read_excel("Db_aout.xlsx")
df=df.dropna()
df = df.sample(frac = 1)
columns=[ 'S0_F0', 'S1_F0', 'S2_F0', 'S3_F0',
       'S4_F0', 'S5_F0', 'S6_F0', 'S7_F0', 'S8_F0', 'S9_F0', 'S10_F0',
       'S11_F0', 'S12_F0', 'S13_F0', 'S14_F0', 'S0_F1', 'S0_F2', 'S0_F3',
       'S0_F4', 'S1_F1', 'S1_F2', 'S1_F3', 'S1_F4', 'S2_F1', 'S2_F2', 'S2_F3',
       'S2_F4', 'S3_F1', 'S3_F2', 'S3_F3', 'S3_F4', 'S4_F1', 'S4_F2', 'S4_F3',
       'S4_F4', 'S5_F1', 'S5_F2', 'S5_F3', 'S5_F4', 'S6_F1', 'S6_F2', 'S6_F3',
       'S6_F4', 'S7_F1', 'S7_F2', 'S7_F3', 'S7_F4', 'S8_F1', 'S8_F2', 'S8_F3',
       'S8_F4', 'S9_F1', 'S9_F2', 'S9_F3', 'S9_F4', 'S10_F1', 'S10_F2',
       'S10_F3', 'S10_F4', 'S11_F1', 'S11_F2', 'S11_F3', 'S11_F4', 'S12_F1',
       'S12_F2', 'S12_F3', 'S12_F4', 'S13_F1', 'S13_F2', 'S13_F3', 'S13_F4',
       'S14_F1', 'S14_F2', 'S14_F3', 'S14_F4', 'S0_F5', 'S1_F5', 'S2_F5',
       'S3_F5', 'S4_F5', 'S5_F5', 'S6_F5', 'S7_F5', 'S8_F5', 'S9_F5', 'S10_F5',
       'S11_F5', 'S12_F5', 'S13_F5', 'S14_F5']

st.dataframe(df)

st.markdown("Notre base de données est constitué de 90 variables explicatives et deux variables cibles. ")
st.markdown("On donne le choix à l'utilisateur entre traiter directement les deux variables cibles ou non ")
cible=st.radio(
              "voulez vouz traiter directement les deux variables cibles ? ",
              ('Oui','Non'),key=100000000000)
if cible=='Oui':
       
       X_train, X_test = train_test_split(df[columns], test_size=0.2, random_state=42, shuffle=True)
       trainScaler=StandardScaler()

       y_train, y_test = train_test_split(df[['ref1', 'ref2']], test_size=0.2, random_state=42, shuffle=True)
       trainScaler=StandardScaler()


       X_trainScaler=StandardScaler()
       X_testScaler=StandardScaler()
       y_trainScaler=StandardScaler()
       y_testScaler=StandardScaler()



       X_trainScalerFit=X_trainScaler.fit(X_train)
       X_testScalerFit=X_testScaler.fit(X_test)
       y_trainScalerFit=y_trainScaler.fit(y_train)
       y_testScalerFit=y_testScaler.fit(y_test)


       X_train=X_trainScalerFit.transform(X_train)
       X_test=X_testScalerFit.transform(X_test)
       y_train=y_trainScalerFit.transform(y_train)
       y_test=y_testScalerFit.transform(y_test)
       
       
       #################
       
if cible=='Non':
       cibles=st.radio(
              "quelle est la variable cible que vous voulez!, ",
              ('première','deuxième'),key=100000000001)
       
       
       if cibles=='première':
              col='ref1'
       else:
              col='ref2'
       
       X=df[columns].values
       Y=df[col].values.reshape(-1, 1)
   


       PredictorScaler=StandardScaler()
       TargetVarScaler=StandardScaler()


       PredictorScalerFit=PredictorScaler.fit(X)
       TargetVarScalerFit=TargetVarScaler.fit(Y)


       X=PredictorScalerFit.transform(X)
       Y=TargetVarScalerFit.transform(Y)



       X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
       
st.subsubheader('Traitement de la base de donnée')      
st.markdown("On commence par standariser les données. ")   
st.latex(r'''
     X_{sta}=\frac{X-\mathbb{E}[X]}{sd[X]}
     ''')

 st.subsubheader('Le modèle')   
st.markdown("On passe à la construction du modèle. ")
image = Image.open('representation-neural-network.webp')
st.image(image, caption='Représentation visuelle d’un réseau de neurones')

st.markdown('Un réseau de neurones est un modèle à plusieurs niveaux inspiré du cerveau humain. Comme les neurones de notre cerveau, les cercles ci-dessus représentent un nœud. Les cercles bleus représentent la couche d’entrée (input layers), les cercles noirs représentent les couches cachées (hidden layers) ou intermédiaires et les cercles verts représentent la couche de sortie (output layers). Chaque nœud des couches cachées représente une fonction qui s’applique sur les données d’entrée menant finalement à une sortie des cercles verts.')

st.markdown(" Veuillez choisir le nombre de couches internes de votre modèle: ")
length=st.slider("Votre choix", step= 2,min_value=0, max_value=50,value= 50) 
functions=['tanh','softmax','relu','softplus','softplus','hard_sigmoid','linear']
units=[]
activation=[]

continuer=True

       
       

for i in range(length):
       unit=st.slider(label="Nombre de neurones", step= 2,min_value=0, max_value=1000,value= 200,key=i)
       
       function=st.radio(
     "quelle est la fonction d'activation? ",
     ('tanh','softmax','relu','softplus','softplus','hard_sigmoid','linear'),key=length+i)
       units.append(unit)
       activation.append(function)

st.write(units,activation)      
#df=df=pd.read_csv('DB aout.csv')



class reg_model():
       
       def __init__(self,Input_shape,Output_shape):
              
              self.model=sequential()
              self.output_shape=Output_shape
              self.input_shape=Input_shape
              
              model.add(Dense(1000, input_shape=(Input_shape,), activation='relu'))
              model.add(Dense(Output_shape, activation='linear'))
              
              
       
       def add_layer(self,units,activation):
              
              last_layer=self.model.layers[-1]
              x=Sequential()

              for layer in self.model.layers[:-1]: 
                       x.add(layer)

              x.add(Dense(units, activation=activation))
              x.add(last_layer) 
              self.model=x
       
       
       def add_layer(self,units,activation):
              
              
              last_layer=self.model.layers[-1]
              x=sequential()
              
              for layer in self.model.layers[:-1]: 
                     x.add(layer)
                     
              x.add(Dense(units, activation=activation))
              x.add(last_layer) 
              self.model=x
                
                     
        
       def compile_model(self):
              self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
         
       
       
       def train_model(self,X_train, y_train,X_test, y_test,epochs):
              
              es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=50,
                   restore_best_weights = True)


              history = self.model.fit(X_train, y_train,
                                  validation_data = (X_test, y_test),
                                  callbacks=[es],
                                  epochs=epochs,
                                  batch_size=50,
                                  verbose=1)
              
              return(history)
       
       
       
       def epoch_vs_losses(history):          
              history_dict = history.history
              loss_values = history_dict['loss'] 
              val_loss_values = history_dict['val_loss']
              epochs = range(1, len(loss_values) + 1) 

       
       
       
                     
                     
 
