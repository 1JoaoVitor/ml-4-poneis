# %%
import pandas as pd 

df = pd.read_excel("../data/dados_cerveja_nota.xlsx")
df
# %%
df['Aprovado'] = df['nota'] >= 5
df 

features = ['cerveja']
target = 'Aprovado' 

# acurácia = acertos/total
# precisão = positivos/vp+fp  (verdadeiro positivo e falso positivo)
# recall ou sensibilidade = positivos/vp+fn
# especificidade = vn/vn+fp (ao contrário do recall)

# %%
# testar primeiro com regressão logística
from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)

#aqui o modelo aprende 
reg.fit(df[features], df[target]) #observado

# aqui o modelo preve 
reg_predict = reg.predict(df[features]) #previsto
reg_predict
# %%
from sklearn import metrics

reg_acc = metrics.accuracy_score(df[target], reg_predict) #comparando o observado com o previsto
reg_acc
print("Acurácia: ", reg_acc)

reg_precision = metrics.precision_score(df[target], reg_predict) 
print("Precisão: ", reg_precision)

reg_recall = metrics.recall_score(df[target], reg_predict) 
print("Recall: ", reg_recall)
# %%
# matriz de confusão, monstra os falsos positivos, falsos negativos, etc. 
# as colunas são o previsto e as linhas o observado
reg_conf = metrics.confusion_matrix(df[target], reg_predict)
reg_conf = pd.DataFrame(reg_conf, index=['False', 'True']
                        , columns=['False', 'True'])
reg_conf
# %%
# testando agora com árvore
from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=2)

#aqui o modelo aprende 
arvore.fit(df[features], df[target]) #observado

# aqui o modelo preve 
arvore_predict = arvore.predict(df[features]) #previsto
arvore_predict
# %%
arvore_acc = metrics.accuracy_score(df[target], arvore_predict)  # arvore aprendeu melhor
print("Acurácia: ", arvore_acc)
arvore_precision = metrics.precision_score(df[target], arvore_predict) 
print("Precisão: ", arvore_precision)
arvore_recall = metrics.recall_score(df[target], arvore_predict) 
print("Recall: ", arvore_recall)
# %%
arvore_conf = metrics.confusion_matrix(df[target], arvore_predict)
arvore_conf = pd.DataFrame(arvore_conf, index=['False', 'True']
                        , columns=['False', 'True'])
arvore_conf
# %%

# por fim o naive bayes 
from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()

#aqui o modelo aprende 
nb.fit(df[features], df[target]) #observado

# aqui o modelo preve 
nb_predict = nb.predict(df[features]) #previsto
nb_predict
# %%
nb_acc = metrics.accuracy_score(df[target], nb_predict) # nb deu a mesma que a regressão logística
print("Acurácia: ", nb_acc)
nb_precision = metrics.precision_score(df[target], nb_predict) 
print("Precisão: ", nb_precision)
nb_recall = metrics.recall_score(df[target], nb_predict) 
print("Recall: ", nb_recall)
# %%
nb_conf = metrics.confusion_matrix(df[target], nb_predict)
nb_conf = pd.DataFrame(nb_conf, index=['False', 'True']
                        , columns=['False', 'True'])
nb_conf
# %%

# o modelo por si só faz o corte na probabilidade  em 0.5 na hora de fazer a predição
# mas podemos mudar esse valor 

nb_proba = nb.predict_proba(df[features])[:,1] 
# retorna uma lista com a probabilidade de cada item em ser reprovado e aprovado
# nesse caso estamos pegando só as probabilidades de ser aprovado 
nb_proba
# %%
# agora vamos fazer as métricas a partir disso 
nb_predict = nb_proba > 0.5 # se manter 0.5 as métricas continuam

nb_acc = metrics.accuracy_score(df[target], nb_predict) # nb deu a mesma que a regressão logística
print("Acurácia: ", nb_acc)
nb_precision = metrics.precision_score(df[target], nb_predict) 
print("Precisão: ", nb_precision)
nb_recall = metrics.recall_score(df[target], nb_predict) 
print("Recall: ", nb_recall)
# %%
nb_conf = metrics.confusion_matrix(df[target], nb_predict)
nb_conf = pd.DataFrame(nb_conf, index=['False', 'True']
                        , columns=['False', 'True'])
nb_conf
# %%
arvore_proba = arvore.predict_proba(df[features])[:,1] 
arvore_proba
# %% 
arvore_predict = arvore_proba > 0.32 # se manter 0.5 as métricas continuam

arvore_acc = metrics.accuracy_score(df[target], arvore_predict) 
print("Acurácia: ", arvore_acc)
arvore_precision = metrics.precision_score(df[target], arvore_predict) 
print("Precisão: ", arvore_precision)
arvore_recall = metrics.recall_score(df[target], arvore_predict) 
print("Recall: ", arvore_recall)
# %%
arvore_conf = metrics.confusion_matrix(df[target], arvore_predict)
arvore_conf = pd.DataFrame(arvore_conf, index=['False', 'True']
                        , columns=['False', 'True'])
arvore_conf
# %%
# curva ROC, observa o recall e especificidade em cada corte de prob e faz um gráfico de curva a partir de cada ponto
roc_curve = metrics.roc_curve(df[target], nb_proba)
roc_curve # retorna os valores de recall, espec

import matplotlib.pyplot as plt 

plt.plot(roc_curve[0], roc_curve[1])
plt.show()
# %%
roc_auc = metrics.roc_auc_score(df[target], nb_proba)
roc_auc # área de baixo da curva, maior melhor

# %%
