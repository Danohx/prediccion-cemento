from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el paquete entrenado
paquete = joblib.load('modelo_regre.pkl')

modelo = paquete['modelo']
escalador = paquete['escalador']
features = paquete['features']
columnas_originales = paquete['columnas_originales']


@app.route('/')
def home():
    return render_template('formulario.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir datos del formulario
        datos = {
            'Cement': float(request.form['cement']),
            'Blast Furnace Slag': float(request.form['slag']),
            'Fly Ash': float(request.form['fly_ash']),
            'Water': float(request.form['water']),
            'Superplasticizer': float(request.form['superplasticizer']),
            'Coarse Aggregate': float(request.form['coarse_aggregate']),
            'Fine Aggregate': float(request.form['fine_aggregate']),
            'Age (day)': float(request.form['age'])
        }

        data_df = pd.DataFrame([datos])

        data_df = data_df[columnas_originales]

        data_escalada = escalador.transform(data_df)

        data_escalada = pd.DataFrame(
            data_escalada,
            columns=columnas_originales
        )

        data_final = data_escalada[features]

        prediccion = modelo.predict(data_final)[0]

        return render_template(
            'formulario.html',
            resultado=f'La resistencia estimada del concreto es: {prediccion:.2f} MPa'
        )

    except Exception as e:
        return render_template(
            'formulario.html',
            resultado=f'Error en la predicción: {str(e)}'
        )


if __name__ == '__main__':
    app.run(debug=True)