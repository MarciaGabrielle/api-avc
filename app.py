import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
from openai import OpenAI

# 🔹 Inicializar API Flask
app = Flask(__name__)

# 🔹 Carregar modelo treinado
modelo = joblib.load("random_forest_model.pkl")

# 🔹 Criar LabelEncoders com as categorias usadas no treinamento
label_encoders = {
    'gender': LabelEncoder().fit(['Male', 'Female']),
    'ever_married': LabelEncoder().fit(['No', 'Yes']),
    'work_type': LabelEncoder().fit(['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed']),
    'Residence_type': LabelEncoder().fit(['Rural', 'Urban']),
    'smoking_status': LabelEncoder().fit(['never smoked', 'formerly smoked', 'smokes', 'Unknown'])
}

# 🔹 Criar normalizador MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# 🔹 Configurar a chave da API da OpenAI
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_recommendations(input_data, prediction):
    """ Função que gera recomendações personalizadas com base na predição """
    
    # 🔹 Formatar os dados do paciente para exibição no prompt
    formatted_data = (
        f"Gênero: {input_data['gender']}, Idade: {input_data['age']}, "
        f"Hipertensão: {'Sim' if input_data['hypertension'] else 'Não'}, "
        f"Doença Cardíaca: {'Sim' if input_data['heart_disease'] else 'Não'}, "
        f"Já foi casado(a): {input_data['ever_married']}, "
        f"Tipo de trabalho: {input_data['work_type']}, "
        f"Tipo de residência: {input_data['Residence_type']}, "
        f"Nível de glicose: {input_data['avg_glucose_level']}, "
        f"IMC: {input_data['bmi']}, "
        f"Status de fumante: {input_data['smoking_status']}"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": (
                "Você é um assistente médico especializado na prevenção de AVCs. "
                "Com base nos dados do paciente, forneça recomendações personalizadas para prevenir um AVC. "
                "Limite sua resposta a no máximo 400 caracteres e forneça as informações no formato de lista."
            )},
            {"role": "user", "content": (
                f"Paciente com os seguintes dados: {formatted_data}. "
                f"Predição de AVC: {prediction}. "
                "Quais são as recomendações diretas e personalizadas para essa pessoa prevenir um possível AVC?"
            )}
        ],
        max_tokens=400
    )
    
    recommendations = response.choices[0].message.content.strip()
    return recommendations

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔹 Receber os dados enviados pelo Power Apps
        dados = request.get_json()

        # 🔹 Criar dicionário com os dados de entrada
        input_data = {
            'gender': dados["genero"],
            'age': int(dados["idade"]),
            'hypertension': int(dados["hipertensao"]),
            'heart_disease': int(dados["doenca_cardiaca"]),
            'ever_married': dados["casado"],
            'work_type': dados["tipo_trabalho"],
            'Residence_type': dados["tipo_residencia"],
            'avg_glucose_level': float(dados["glicose"]),
            'bmi': float(dados["imc"]),
            'smoking_status': dados["fumante"] if dados["fumante"] else 'Unknown'
        }

        # 🔹 Mapeamento dos valores para os valores esperados pelo LabelEncoder
        gender_map = {'Masculino': 'Male', 'Feminino': 'Female'}
        married_map = {'Sim': 'Yes', 'Não': 'No'}
        work_type_map = {
            'Crianças': 'children',
            'Emprego Governamental': 'Govt_job',
            'Nunca Trabalhou': 'Never_worked',
            'Privado': 'Private',
            'Autônomo': 'Self-employed'
        }
        residence_map = {'Rural': 'Rural', 'Urbano': 'Urban'}
        smoking_map = {
            'Nunca fumou': 'never smoked',
            'Fumou': 'formerly smoked',
            'Fuma': 'smokes',
            'Desconhecido': 'Unknown'
        }

        # 🔹 Aplicar o mapeamento nos valores recebidos
        input_data['gender'] = gender_map[input_data['gender']]
        input_data['ever_married'] = married_map[input_data['ever_married']]
        input_data['work_type'] = work_type_map[input_data['work_type']]
        input_data['Residence_type'] = residence_map[input_data['Residence_type']]
        input_data['smoking_status'] = smoking_map[input_data['smoking_status']]

        # 🔹 Criar DataFrame
        df = pd.DataFrame([input_data])

        # 🔹 Aplicar Label Encoding nas colunas categóricas
        for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
            df[col] = label_encoders[col].transform(df[col])

        # 🔹 Fazer a predição com o modelo carregado
        predicao = modelo.predict(df)[0]  # Retorna 0 (Não) ou 1 (Sim)

        # 🔹 Converter para resposta "Sim" ou "Não"
        resultado = "Sim" if predicao == 1 else "Não"

        # 🔹 Gerar recomendação personalizada
        recomendacao = generate_recommendations(input_data, resultado)

        return jsonify({"predicao": resultado, "recomendacao": recomendacao})

    except Exception as e:
        return jsonify({"erro": str(e)})

# 🔹 Rodar o servidor
if __name__ == "__main__":
    app.run(debug=True)
