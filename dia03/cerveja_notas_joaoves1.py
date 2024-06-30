# %%

import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_excel("../data/dados_cerveja_nota.xlsx")

df
# %%
# Criando um gráfico para vizualizar 
plt.plot(df["cerveja"], df["nota"], 'o') 
plt.grid(True)
plt.title("Relação nota cerveja ")
plt.xlabel("Cerveja")
plt.ylabel("Notas")
plt.show()
# %%
# criando a reta da regressão (como visto na aula 2 de ML)
from sklearn import linear_model

reg = linear_model.LinearRegression() # mesma lógica da arvore

reg.fit(df[["cerveja"]], df["nota"]) # aqui o primeiro tem que ser uma matriz, por isso dois []
# %%
reg.coef_ # o "b" da reta y = a + bx, retorna uma lista de coef 
# %%
reg.intercept_ # o "a" da reta y = a + bx
# %%
x = df[["cerveja"]].drop_duplicates()
y_estimado = reg.predict(x) # pega os valores de x e aplica na formula com o a e b
y_estimado

plt.plot(df["cerveja"], df["nota"], 'o') 
plt.plot(x, y_estimado, '-') # fazer a reta
plt.grid(True)
plt.title("Relação nota cerveja ")
plt.xlabel("Cerveja")
plt.ylabel("Notas")
plt.show()
# %%

from sklearn import tree 

arvore = tree.DecisionTreeRegressor(max_depth=2) # de regressão agora

arvore.fit(df[["cerveja"]], df["nota"])
# %%


y_estimado_arvore = arvore.predict(x) # pega os valores de x e aplica na formula com o a e b
y_estimado_arvore

# %%

plt.plot(df["cerveja"], df["nota"], 'o') 
plt.plot(x, y_estimado, '-') # fazer a reta
plt.plot(x, y_estimado_arvore, '-') # plotando a árvore (ideal ter uma vizualização em escada, isso se modifica com max_deph)
plt.grid(True)
plt.title("Relação nota cerveja ")
plt.xlabel("Cerveja")
plt.ylabel("Notas")
plt.legend(["Pontos", "Regressão", "Árvore"])
plt.show()
# %%
