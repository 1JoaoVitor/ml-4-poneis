# %% 
import pandas as pd 
import sqlalchemy
model = pd.read_pickle("modelo_rf_joaoves1.pkl") # carregando o modelo salvo, ou seja, separando treino e uso dele para prever
model
# %%

# SUPONDO que os dados que usamos para treino tbm serão dados novos (não acontece normalmente)

dt = '2024-05-08' # definindo a data que vai executar

with open("churn_model/etl_model.sql", "r") as open_file: # vai no banco de dados, executa a query e traz os dados "novos" de lá
    query = open_file.read()

query = query.format(date=dt)

engine = sqlalchemy.create_engine("sqlite:///../../data/database_upsell.db") # conexão com o bd

# %%
df = pd.read_sql_query(query, engine)
df
# %%
proba = model['model'].predict_proba(df[model['features']])[:,1]
proba


df['proba_active'] = proba
(df[['Name', 'proba_active']].sort_values(by='proba_active', ascending=False)
                             .head(25)
 )
# faz de fato a predição usando dados novos
# %%
