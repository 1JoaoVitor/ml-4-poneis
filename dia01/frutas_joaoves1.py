# %% 
import pandas as pd 

df = pd.read_excel("../data/dados_frutas.xlsx")
df
# %%

filtro_arredondada = df["Arredondada"] == 1

df[filtro_arredondada]
# %%

filtro_suculenta = df["Suculenta"] == 1

df[filtro_arredondada & filtro_suculenta]
# %%

filtro_vermelha = df["Vermelha"] == 1

filtro_doce = df["Doce"] == 1

df[filtro_arredondada & filtro_suculenta & filtro_vermelha & filtro_doce]
# %%

from sklearn import tree # biblioteca de ML

features = ["Arredondada", "Suculenta", "Vermelha", "Doce"] # atributos/variaveis

target = "Fruta" #variavel alvo 

x = df[features] # normalemente X são os atributos 
y = df[target] # normalmente y são os alvos
# %%

arvore = tree.DecisionTreeClassifier()
arvore.fit(x, y) # método fit = aprenda 
# %%

tree.plot_tree(arvore, 
               class_names=arvore.classes_, 
               feature_names=features,
               filled=True) # nós pintados nesse caso mostram nós puros, que só tem um tipo de fruta
# %%

# até agora separamos em grupos, mas como prever novos valores?
# valores ["Arredondada", "Suculenta", "Vermelha", "Doce"]  
arvore.predict([[1, 1, 1, 1]]) 

#predict te da a informação direta
# %%

probas = arvore.predict_proba([[1, 1, 1, 1]])[0] # te da uma lista de listas, só pegando a primeira
# predict_proba retorna a probabilidade de cada label (fruta)
# %%

pd.Series(probas, index=arvore.classes_)
# %%
