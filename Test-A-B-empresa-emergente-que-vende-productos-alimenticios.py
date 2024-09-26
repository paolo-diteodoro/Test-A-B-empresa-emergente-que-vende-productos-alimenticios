#!/usr/bin/env python
# coding: utf-8

# 
# 
# **Hola! Paolo**
# 
# Mi nombre es Enrique Romero, tendre el gusto de revisar Tu proyecto, te deseo lo mejor.
# 
# <div class="alert alert-danger">
# <b>❌ Comentario del revisor:</b> Esto destaca los comentarios más importantes. Sin su desarrollo, el proyecto no será aceptado. </div>
# 
# <div class="alert alert-warning">
# <b>⚠️ Comentario del revisor:</b> Así que los pequeños comentarios están resaltados. Se aceptan uno o dos comentarios de este tipo en el borrador, pero si hay más, deberá hacer las correcciones. Es como una tarea de prueba al solicitar un trabajo: muchos pequeños errores pueden hacer que un candidato sea rechazado.
# </div>
# 
# <div class="alert alert-success">
# <b>✔️ Comentario del revisor:</b> Así que destaco todos los demás comentarios.</div>
# 
# <div class="alert alert-info"> <b>Comentario del estudiante:</b> Por ejemplo, asi.</div>
# 
# Todo esto ayudará a volver a revisar tu proyecto más rápido.
# 
# 
# 
# 

# # ***Introducción al Proyecto***
# 
# Trabajas en una empresa emergente que vende productos alimenticios. Debes investigar el comportamiento del usuario para la aplicación de la empresa.
# 
# Primero, estudia el embudo de ventas. Descubre cómo los usuarios y las usuarias llegan a la etapa de compra. ¿Cuántos usuarios o usuarias realmente llegan a esta etapa? ¿Cuántos se atascan en etapas anteriores? ¿Qué etapas en particular?
# 
# Luego, observa los resultados de un test A/A/B. (Sigue leyendo para obtener más información sobre los test A/A/B). Al equipo de diseño le gustaría cambiar las fuentes de toda la aplicación, pero la gerencia teme que los usuarios y las usuarias piensen que el nuevo diseño es intimidante. Por ello, deciden tomar una decisión basada en los resultados de un test A/A/B.
# 
# Los usuarios se dividen en tres grupos: dos grupos de control obtienen las fuentes antiguas y un grupo de prueba obtiene las nuevas. Descubre qué conjunto de fuentes produce mejores resultados.
# 
# Crear dos grupos A tiene ciertas ventajas. Podemos establecer el principio de que solo confiaremos en la exactitud de nuestras pruebas cuando los dos grupos de control sean similares. Si hay diferencias significativas entre los grupos A, esto puede ayudarnos a descubrir factores que pueden estar distorsionando los resultados. La comparación de grupos de control también nos dice cuánto tiempo y datos necesitaremos cuando realicemos más tests.
# 
# Utilizarás el mismo dataset para el análisis general y para el análisis A/A/B. En proyectos reales, los experimentos se llevan a cabo constantemente. El equipo de análisis estudia la calidad de una aplicación utilizando datos generales, sin prestar atención a si los usuarios y las usuarias participan en experimentos.

# ## Descargar los datos e importar libreria
# 
# - Importamos librerias

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# - Cargamos los datos

# In[2]:


data = pd.read_csv("/datasets/logs_exp_us.csv", sep="	")


# ##  Preparar los datos para el análisis
# ### Veamos las columnas, tipos de datos, valores ausentes y duplicados

# In[3]:


data.head()


# - Debemos cambiar los titulos de las columnas y colocarlos todos en minuscula. El formato de fecha debemos modificarlo

# In[4]:


data.info()


# In[5]:


data.isna().sum()


# - No vemos valores nulos

# In[6]:


data_1 = pd.DataFrame(data)

duplicados = data.duplicated().sum()

duplicados


# - Tenemos 413 valores duplicados, seguramente en la columna Event Name y Expld

# ### Preparar los datos para el analisis

# - Cambiamos los titulos de las columnas para que sean todos minuscula y colocamos "_" a cambio de espacios

# In[7]:


data.columns = data.columns.str.lower()


# In[8]:


data.rename(
    columns={'eventname': 'event_name', 
             'deviceidhash': 'device_id_hash', 
             'eventtimestamp': 'event_time_stamp',
            'expid': 'exp_id'}, 
    inplace=True)


# In[9]:


data.head()


# - Cambiamos formato de fecha de la columna "event_time_stamp" y agregamos la columna de fecha con hora y otra solo con la fecha

# In[10]:


data["date_hour"] = pd.to_datetime(data["event_time_stamp"],unit="s")
data["date"] = pd.to_datetime(data["date_hour"]).dt.to_period("D")


# In[11]:


data.head()


# 
# 
# 
# 
# <div class="alert alert-success">
# <b>✔️ Comentario del revisor:</b>    
# Muy buen trabajo con la carga de los datos y la exploracion inicial
# </div>	
# 

# ## Estudiar y comprobar los datos

# In[12]:


print(f"""
-Número de eventos totales: {len(data):,}

-Eventos por tipo de evento: {data.groupby("event_name")["device_id_hash"].count()}

-Usuarios únicos: {data["device_id_hash"].nunique():,}

-Promedio de eventos por usuario: {round(len(data) / data["device_id_hash"].nunique(), 4):,}
""")


# In[13]:


print(f"""
Fecha mínima: {data["date"].min()}
Fecha máxima: {data["date"].max()}
""")


# In[14]:


plt.figure(figsize=(20, 10))
sns.histplot(data["date_hour"])
plt.xticks(rotation=90)
plt.show()


# In[15]:


filtered_data = data.loc[data["date_hour"] >= "2019-08-01"]


# - Poemos excluir los datos anteriores al 01/08/2019

# In[16]:


plt.figure(figsize=(20, 10))
sns.histplot(filtered_data["date_hour"])
plt.xticks(rotation=90)
plt.show()


# In[17]:


print(f"""
Porcentaje de usuarios que permanecen: {100*round(filtered_data["device_id_hash"].nunique() / data["device_id_hash"].nunique(), 3)}
Porcentaje de usuarios que eliminamos: {100*round(1 - filtered_data["device_id_hash"].nunique() / data["device_id_hash"].nunique(), 3)}
""")


# - Permanece un 99% de los usuarios

# In[18]:


filtered_data.groupby("exp_id")["device_id_hash"].count()


# - Verificamos que tenemos usuarios para los 3 grupos del experimento

# 
# 
# 
# 
# <div class="alert alert-success">
# <b>✔️ Comentario del revisor:</b>    
# Las graficas y el diseño esta bien aplicado, buen trabajo
# </div>	
# 

# ## Estudiar el embudo de eventos
# 

# In[19]:


plt.figure(figsize=(8, 4))
filtered_data.pivot_table(
    index="event_name",
    values="device_id_hash",
    aggfunc="count"
).sort_values(by="device_id_hash", ascending=False).plot(kind="bar", ax=plt.gca())
plt.show()

data.groupby("event_name")["device_id_hash"].count().sort_values(ascending=False)


# - El mayor numero de eventos corresponde a MainScreenAppear           

# In[20]:


filtered_data.groupby("event_name")["device_id_hash"].nunique().sort_values(ascending=False)


# - El mayor numero de usuarios que realiza un evento tambien corresponde a MainScreenAppear           

# In[21]:


filtered_data.groupby("event_name")["device_id_hash"].nunique() / data["device_id_hash"].nunique()


# - El 98% de los usuarios realizaron la accion MainScreenAppear al menos una vez, y solo el 11% reviso el tutorial

# In[22]:


import plotly.express as px

events_by_group = filtered_data.groupby(['exp_id', 'event_name']).agg({"device_id_hash": ["count", "nunique"]}).unstack(fill_value=0)
events_by_group = events_by_group.reset_index()
events_by_group['exp_id'] = events_by_group['exp_id'].astype(str)

events_by_group

# events_by_group = events_by_group.set_index('group').apply(lambda x: x.sort_values(ascending=False),
#                                                            axis=1).reset_index()



# In[23]:


plot_data = (events_by_group
             .melt(id_vars=[('exp_id', '', '')], value_vars=[c for c in events_by_group.columns if "count" in c])
             .rename(columns={"variable_2": "event_name", ('exp_id', '', ''): "exp_id"})
            )

color_map = {'246': 'gold', '247': 'coral', '248': 'steelblue'}
fig = px.bar(plot_data,
             x='event_name', y='value', color="exp_id",
             labels={'event_name': 'Tipo de evento', 'value': 'Frecuencia'},
             title='Frecuencia de eventos por grupo',
             barmode='group',
             color_discrete_map=color_map,
             height=450, width=950
            )

fig.show()

plot_data = (
    events_by_group
    .melt(id_vars=[('exp_id', '', '')], value_vars=[c for c in events_by_group.columns if "nunique" in c])
    .rename(columns={"variable_2": "event_name", ('exp_id', '', ''): "exp_id"})
)

color_map = {'246': 'gold', '247': 'coral', '248': 'steelblue'}
fig = px.bar(plot_data,
             x='event_name', y='value', color="exp_id",
             labels={'event_name': 'Tipo de evento', 'value': 'Frecuencia'},
             title='Usuarios únicos por grupo y evento',
             barmode='group',
             color_discrete_map=color_map,
             height=450, width=950
            )

fig.show()


# 
# 
# 
# <div class="alert alert-success">
# <b>✔️ Comentario del revisor:</b> Los calculos son correctos.</div>
# 

# - Dado el contexto del negocio el orden de las acciones puede ser: 
# 
# 1) Main Screen Appear
# 2) Offers Screen Appear
# 3) Cart Screen Appear
# 4) Payment Screen Successful
# 
# La accion de tutorial no consideramos que sea de la secuencia

# In[24]:


project_funnel_df = filtered_data.query("event_name != 'Tutorial'").pivot_table(
    index="event_name",
    values="device_id_hash",
    aggfunc="nunique"
)
project_funnel_df = project_funnel_df.sort_values(by="device_id_hash", ascending=False).reset_index()
project_funnel_df


# In[25]:


project_funnel = project_funnel_df.sort_values(by="device_id_hash", ascending=False)
project_funnel["users_in_previous_step"] = project_funnel["device_id_hash"].shift(1)
project_funnel["conversion_from_previous_step"] = project_funnel["device_id_hash"] / project_funnel["users_in_previous_step"]
project_funnel["total_conversion"] = project_funnel["device_id_hash"] / max(project_funnel["device_id_hash"])
project_funnel


# In[26]:


import plotly.express as px

fig = px.funnel(project_funnel_df, x='device_id_hash', y='event_name')

fig.show()


# - En la etapa de Offers Screen Appear es donde se pierden mas usuarios con respecto al Main Screen Appear, casi un 40%. Casi un 50% de los usuarios hace todo el viaje desde la primera etapa hasta el pago.

# ## Estudiar los resultados del experimento

# In[27]:


n_usuarios = filtered_data.query("event_name != 'Tutorial'").groupby("exp_id")["device_id_hash"].nunique()
n_usuarios


# - En cada grupo vemos que tenemos casi el mismo numero de usuarios, entorno a los 2500

# In[28]:


conversion = filtered_data[['exp_id', 'device_id_hash']].drop_duplicates()
conversion.head()


# In[29]:


converted = pd.DataFrame(data={
    "device_id_hash": filtered_data[filtered_data["event_name"] == "PaymentScreenSuccessful"]["device_id_hash"].unique(),
    "converted": 1
})

converted.head()


# In[30]:


conversiones = conversion.merge(converted, on="device_id_hash", how="left")

conversiones["converted"] = conversiones["converted"].fillna(0)

conversiones.head()


# In[31]:


usuario_246 = conversiones[conversiones["exp_id"] == 246]["converted"]
usuario_247 = conversiones[conversiones["exp_id"] == 247]["converted"]


# In[32]:


from scipy.stats import ttest_ind

statistic, pvalue = ttest_ind(
    usuario_246,
    usuario_247
)

print(f"""
Statistic: {statistic}
p-value: {pvalue}
""")


# - Realizado el test y dado que el valor p-value es mayor que un nivel de significancia comúnmente elegido (como 0.05), no hay suficiente evidencia para rechazar la hipótesis nula (las tasas promedio de conversion de los grupos de control son iguales). Por lo tanto, no podemos concluir que hay una diferencia significativa entre las medias de usuario_246 y usuario_247.

# In[33]:


filtered_data[filtered_data["exp_id"] == 246]["event_name"].value_counts().sort_values(ascending=False)


# In[34]:


experiment_246 = filtered_data[filtered_data["exp_id"] == 246].groupby('event_name').agg({"device_id_hash": ["nunique"]}).sort_values(by=('device_id_hash', 'nunique'), ascending=False).reset_index()
experiment_246


# In[35]:


experiment_246["device_id_hash"] / experiment_246["device_id_hash"].sum()


# In[36]:


filtered_data[filtered_data["exp_id"] == 247]["event_name"].value_counts().sort_values(ascending=False)


# In[37]:


experiment_247 = filtered_data[filtered_data["exp_id"] == 247].groupby('event_name').agg({"device_id_hash": ["nunique"]}).sort_values(by=('device_id_hash', 'nunique'), ascending=False).reset_index()
experiment_247


# In[38]:


experiment_247["device_id_hash"] / experiment_247["device_id_hash"].sum()


# In[39]:


filtered_data[filtered_data["exp_id"] == 248]["event_name"].value_counts().sort_values(ascending=False)


# In[40]:


experiment_248 = filtered_data[filtered_data["exp_id"] == 248].groupby('event_name').agg({"device_id_hash": ["nunique"]}).sort_values(by=('device_id_hash', 'nunique'), ascending=False).reset_index()
experiment_248


# In[41]:


experiment_248["device_id_hash"] / experiment_248["device_id_hash"].sum()


# - El evento mas popular es los dos grupos de control es el MainScreenAppear. Los usuarios que realizaron esta accion en el grupo de control 246 fueron 2450 con una proporcion de 36% del total y en el grupo de control 247 fueron  2476 con una proporcion de 37% del total.

# In[42]:


def t_test(df, group1, group2, event, group_col="exp_id"):
    
    conversions = df[[group_col, "device_id_hash"]].drop_duplicates()
    
    converted = pd.DataFrame(data={
      "device_id_hash": df[df["event_name"] == event]["device_id_hash"].unique(),
      "converted": 1
    })
    
    conversions = conversions.merge(converted, on="device_id_hash", how="left")
    
    conversions["converted"] = conversions["converted"].fillna(0)
    
    statistic, pvalue = ttest_ind(
        conversions[conversions[group_col] == group1]["converted"],
        conversions[conversions[group_col] == group2]["converted"]
    )
    
    return statistic, pvalue


# In[43]:


events = ["MainScreenAppear", "OffersScreenAppear", "CartScreenAppear", "PaymentScreenSuccessful"]
group1 = 246
group2 = 247

print("Running A/A tests...")

for event in events:
    
    pvalue = t_test(filtered_data, group1, group2, event)
    
    print(f"""
    Event: {event}
    p-value: {pvalue}
    """)
    print()


# - En todos los eventos analizados para los grupos de control el valor p es alto (mayor que 0.05), por lo que no hay suficiente evidencia para rechazar la hipótesis nula y se concluye que no hay una diferencia significativa entre los grupos en términos de la frecuencia de ocurrencia del evento. En conclusion podemos decir que los grupos de control de dividieron correctamente.

# <div class="alert alert-success">
# <b>✔️ Comentario del revisor:</b> Asi la evidencia en este caso permite diferenciar la tendencia de los valores en base a la metrica de la distribucion normal.</div>
# 

# In[44]:


data_control = filtered_data.copy()

data_control.loc[data_control["exp_id"] == 246, "exp_id"] = 'control'
data_control.loc[data_control["exp_id"] == 247, "exp_id"] = 'control'

data_control


# - Hemos creado un nuevo dataframe en donde se combinan los gruspo de control 246 y 247 para asi poder hacer posteriomente los t_test del experimiento vs todos los controles agrupados.

# In[45]:


events = ["MainScreenAppear", "OffersScreenAppear", "CartScreenAppear", "PaymentScreenSuccessful"]
controls = [246, 247]


print("Running A/B tests...")
print()
for event in events:

  print(f"Event: {event}")

  for control in controls:

    _, pvalue = t_test(filtered_data, control, 248, event)

    print(f"""
    Comparison: {control} vs. experiment
    p-value: {pvalue}
    """)

  _, pvalue = t_test(data_control, 'control', 248, event, group_col="exp_id")

  print(f"""
  Comparison: all control vs. experiment
  p-value: {pvalue}
  """)


# - En todos los 12 eventos analizados para grupos de control vs experimiento el valor p han sido altos (mayor que 0.05), por lo que no hay suficiente evidencia para rechazar la hipótesis nula y se concluye que no hay una diferencia significativa entre los grupos en términos de la frecuencia de ocurrencia del evento. El test que mas cercano dio al 0.05 fue Event: CartScreenAppear
# en donde Comparison: 246 vs. experiment p-value: 0.07845751752267902.

# In[46]:


n_pruebas = 12
significancia = 0.05

print(f"""
Número de pruebas = {n_pruebas}
alpha = {significancia}
Probabilidad de falso positivo: {1-(1 - significancia)**n_pruebas}
""")

significancia = 0.01

print(f"""
Número de pruebas = {n_pruebas}
alpha = {significancia}
Probabilidad de falso positivo deseada: {round(1-(1 - significancia)**n_pruebas, 6)}
""")


# In[47]:


events = ["MainScreenAppear", "OffersScreenAppear", "CartScreenAppear", "PaymentScreenSuccessful"]
controls = [246, 247]


print("Running A/B tests...")
print()
for event in events:

  print(f"Event: {event}")

  for control in controls:
        
    alpha = 0.001
    _, pvalue = t_test(filtered_data, control, 248, event)

    print(f"""
    Comparison: {control} vs. experiment
    p-value: {pvalue}
    """)

  _, pvalue = t_test(data_control, 'control', 248, event, group_col="exp_id")

  print(f"""
  Comparison: all control vs. experiment
  p-value: {pvalue}
  """)


# <div class="alert alert-danger">
# <b>✔️ Comentario del revisor:</b>Vamos bien solamente faltan las conclusiones finales y el analisis de los argumentos basado en la definicion inicial .</div>
# 
