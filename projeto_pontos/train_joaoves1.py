# %% 
import pandas as pd 
from sklearn import model_selection
df = pd.read_csv("../data/dados_pontos.csv", sep=";")
df
# %%

features = df.columns[3:-1]
target = 'flActive'
# %%

x_train, x_test, y_train, y_test = model_selection.train_test_split(df[features], 
                                                                    df[target],
                                                                    test_size=0.2, #quanto vai para teste
                                                                    random_state=42, #seed de sorteio, para garantir os mesmos dados para meu teste e o de outra pessoa
                                                                    stratify=df[target]) # para deixar as taxas de treino e teste mais parecidas

print("Taxa resposta treino", y_train.mean())
print("Taxa resposta teste", y_test.mean())

# %%

# verificar os dados vazio ou com problemas 

x_train.isna().sum()
# %%
max_avgRecorrencia = df["avgRecorrencia"].max()
x_train["avgRecorrencia"] = x_train["avgRecorrencia"].fillna(max_avgRecorrencia)
x_test["avgRecorrencia"]  = x_test["avgRecorrencia"].fillna(max_avgRecorrencia) # preenche com o mesmo valor que foi descoberto no treino
# os dados de teste são para simular dados novos, não devem ser mexidos ou levados em conta 
# agora sim podemos começar a treinar 
# %%

from sklearn import tree
from sklearn import metrics

# aqui a gente treina
arvore = tree.DecisionTreeClassifier(max_depth=5, random_state=42)
arvore.fit(x_train, y_train) # treino
# %%
# aqui a gente preve na própria base
tree_predict_train = arvore.predict_proba(x_train)[:, -1] > 0.5
acc_arvore_train = metrics.accuracy_score(y_train, tree_predict_train)
print("Acc Arvore train: ", acc_arvore_train)

# aqui a gente preve na base de teste 
tree_predict_test = arvore.predict_proba(x_test)[:, -1] > 0.5
acc_arvore_test = metrics.accuracy_score(y_test, tree_predict_test)
print("Acc Arvore test: ", acc_arvore_test)

# %%
1-y_test.mean() # essa é a probabilidade de "chute" sem o modelo, a acc do modelo tem que estar melhor do que isso
# %%

tree_proba_train = arvore.predict_proba(x_train)[:, -1]
auc_arvore_train = metrics.roc_auc_score(y_train, tree_proba_train)
print("Auc Arvore train: ", auc_arvore_train)

tree_proba_test = arvore.predict_proba(x_test)[:, -1]
auc_arvore_test = metrics.roc_auc_score(y_test, tree_proba_test)
print("Auc Arvore test: ", auc_arvore_test)
# %%

