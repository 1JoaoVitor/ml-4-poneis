# %% 
import pandas as pd

df = pd.read_parquet("../data/dados_clones.parquet", engine= "fastparquet")
df
# %%

df.groupby(["Status "])[['Estatura(cm)', 'Massa(em kilos)']].mean()
# agrupando os clones e observando se altura e massa em média são significativas 
# %%

df['Status_bool'] = df['Status '] == 'Apto'
df

# %%
df.groupby(['Distância Ombro a ombro'])['Status_bool'].mean()
# %%
df.groupby(['Tempo de existência(em meses)'])['Status_bool'].mean()
# %%
df.groupby(['Tamanho dos pés'])['Status_bool'].mean()

# %%
df.groupby(['Tamanho do crânio'])['Status_bool'].mean()
# %%
df.groupby(['General Jedi encarregado'])['Status_bool'].mean()
# opa, algo encontrado nesse, uma variação considerável
# %%

features = ['Massa(em kilos)', 'Estatura(cm)'
            ,'Distância Ombro a ombro', 'Tamanho do crânio', 'Tamanho dos pés']

categ_features = ['Distância Ombro a ombro', 'Tamanho do crânio', 'Tamanho dos pés']

x = df[features]
x
# %%

from feature_engine import encoding
# transformou as variáveis categoricas em numéricas
onehot = encoding.OneHotEncoder(variables=categ_features) 
onehot.fit(x)
x = onehot.transform(x)
x
    # %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(x, df['Status '])
# %%
tree.plot_tree(arvore, class_names=arvore.classes_, feature_names=x.columns, filled=True)
# com isso cnoseguimos separar alguns grupos de aptos e defeituosos, mostrando que
# a relação com os generais não era única e talvez determinante 


# %%
