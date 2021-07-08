import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

#Instanciation d'un objet Flask
app = Flask(__name__)
#Désserialisation du modèle
model = joblib.load("model_bagging.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predire', methods=['POST'])
def predict():
    #Chargement de notre modèle 
    model = joblib.load("model_bagging.pkl")    
    json_ = request.get_json(force=True)
    #Recuperation de la chaine de caractère qui forme les permissions
    permissions_str = json_['permissions']
    #Récupération des permissions
    #Les permissions 0 et 1 sont séparées par des ',' 
    permissions_car = permissions_str.split(',')
    #Récupération sous forme de liste
    permissions_list = [int(x) for x in permissions_car]
    #Transformation de la liste en Array
    permissions = np.array([permissions_list])
    #Passer les permissions au modèle et récupération de la prédiction
    prediction = model.predict(permissions)[0]
    #L'envoie de la prédiction en format JSON
    return jsonify({'prediction':str(prediction)})
        
if __name__ == '__main__':
    app.run()
