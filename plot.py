import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import decomposition
import folium 
import numpy as np

plt.style.use('ggplot')

def save(savename, saving_function):
    parent_abs = os.path.abspath(os.path.join(os.getcwd(),os.pardir))
    parent = 'reports'
    path = os.path.join(parent_abs, parent, savename)
    
    saving_function(path)
    
    print('figure saved on ', path)


def template(savename=False):

    fig = plt.figure(figsize=(10,10))

    if savename:
        save(savename, fig.savefig)
    plt.show()

def corr_map(correlation_matrix, savename=False):
    fig = plt.figure(figsize=(12,10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation matrix")
    
    if savename:
        save(savename,fig.savefig)
    plt.show()

def boxplot(data, title, savename=None, figsize=(8, 6)):

    """
    Create a single boxplot for a specific column.

    Parameters:
    - data (DataFrame): The DataFrame containing the data to be plotted.
    - column (str): The column name for which the boxplot will be created.
    - title (str): The title of the plot.
    - savename (str, optional): If provided, the plot will be saved with this filename.
    - figsize (tuple, optional): The size of the figure. Default is (8, 6).

    Returns:
    - None
    """

    plt.figure(figsize=figsize)
    sns.boxplot(x=data)
    plt.title(title)
    # plt.xlabel(data)
    
    if savename:
        plt.savefig(savename)  
    
    plt.show()


def boxplot_subplots(data, rows, cols, savename=False, figsize=(12, 10)):

    """
    Create subplots for visualizing multiple boxplots.

    Parameters:
    - data (DataFrame): The DataFrame containing the data to be plotted.
    - rows (int): The number of rows of subplots.
    - cols (int): The number of columns of subplots.
    - figsize (tuple, optional): The size of the entire figure. Default is (12, 10).
    """

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    column_names = data.columns

    if rows == 1 and cols == 1:
        axes = [[axes]] 
    
    for i in range(rows):
        for j in range(cols):
            if rows > 1 and cols > 1:
                ax = axes[i][j] 
            elif rows > 1:
                ax = axes[i]  
            elif cols > 1:
                ax = axes[j] 
            else:
                ax = axes  
            
            col_index = i * cols + j
            if col_index < len(column_names):
                sns.boxplot(x=column_names[col_index], data=data, ax=ax)
                ax.set_title(f'Boxplot of {column_names[col_index]}')
            else:
                ax.axis('off') 
    if savename:
        save(savename,fig.savefig)

    plt.tight_layout()
    plt.show()

def distribution(data, title, savename=None, figsize=(8, 6)):
    """
    Create a histogram to visualize the distribution of the data.

    Parameters:
    - data (Series or DataFrame column): The data to be plotted.
    - title (str): The title of the plot.
    - savename (str, optional): If provided, the plot will be saved with this filename.
    - figsize (tuple, optional): The size of the figure. Default is (8, 6).

    Returns:
    - None
    """
    plt.figure(figsize=figsize)
    sns.histplot(data, kde=True, element="step")  
    plt.title(title)
    
    if savename:
        plt.savefig(savename)
    
    plt.show()

def pca_2d(X, savename=False):
    '''
    receives X features
    return a scatter plot of 2 components pca
    and color it with third component
    '''
    fig = plt.figure(figsize=(10,10))

    pca = decomposition.PCA(n_components=3)
    view = pca.fit_transform(X)

    plt.scatter(view[:,0], view[:,1], c=view[:,2])
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')

    if savename :
        save(savename, fig.savefig)

    plt.show()


def pca_3d(X, multiple_graph=False, savename=False):
    '''
    plot a 3 components pca
    if multiple_graph: more 3 2d combinations        
    '''

    pca = decomposition.PCA(n_components=4)
    view = pca.fit_transform(X)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(221, projection='3d')
    alpha = 0.4

    if not multiple_graph:

        ax.scatter(view[:,0], view[:,1], view[:,2], c=view[:,3], alpha=alpha)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()

    if multiple_graph:

        ax.scatter(view[:,0], view[:,1], view[:,2], c=view[:,3], alpha=alpha)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax = fig.add_subplot(222)
        ax.scatter(view[:,0], view[:,1], c=view[:,2], alpha=alpha)
        ax.set_xlabel('y')
        ax.set_ylabel('z')

        ax = fig.add_subplot(223)
        ax.scatter(view[:,0], view[:,2], c=view[:,1], alpha=alpha)
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        ax = fig.add_subplot(224)
        ax.scatter(view[:,1], view[:,2], c=view[:,3], alpha=alpha)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    if savename :
        save(savename, fig.savefig)

    plt.show()

def bubble_map(dataframe, coluna_estado):

    estados = {
        "AC": [-9.0479, -70.5265],
        "AL": [-9.6612, -36.6502],
        "AP": [1.3545, -51.9160],
        "AM": [-3.4653, -62.2159],
        "BA": [-12.9714, -38.5014],
        "CE": [-3.7172, -38.5433],
        "DF": [-15.7754, -47.7979],
        "ES": [-20.3155, -40.3128],
        "GO": [-16.6799, -49.2550],
        "MA": [-2.5387, -44.2827],
        "MT": [-12.6510, -56.5293],
        "MS": [-20.4486, -54.6295],
        "MG": [-18.5760, -44.0350],
        "PA": [-5.7945, -55.0974],
        "PB": [-7.1219, -36.7241],
        "PR": [-25.2521, -52.0215],
        "PE": [-8.0476, -34.8770],
        "PI": [-8.0476, -34.8770],
        "RJ": [-22.9068, -43.1729],
        "RN": [-5.7945, -36.2705],
        "RS": [-30.1097, -51.3174],
        "RO": [-11.5057, -63.5806],
        "RR": [2.7376, -62.0751],
        "SC": [-27.5954, -48.5480],
        "SP": [-23.5505, -46.6333],
        "SE": [-10.9472, -37.0731],
        "TO": [-10.9472, -37.0731]
    }
    
    if coluna_estado not in dataframe.columns:
        raise ValueError("A coluna de estados não existe no DataFrame.")
    
    contagem_estados = dataframe[coluna_estado].value_counts().to_dict()
    
    valores_normalizados = {estado: np.interp(contagem, (min(contagem_estados.values()), max(contagem_estados.values())), (1, 10)) for estado, contagem in contagem_estados.items()}
    
    mapa = folium.Map(location=[-14.235, -51.925], zoom_start=4)
    
    for estado, coordenadas in estados.items():
        if estado in valores_normalizados:
            tamanho_bolha = valores_normalizados[estado]
            folium.CircleMarker(location=coordenadas, radius=tamanho_bolha*3, color='blue', fill=True, fill_color='blue', fill_opacity=0.6, popup=f"{estado}: {contagem_estados[estado]} ocorrências").add_to(mapa)
    
    return mapa
    


def pin_map(dataframe, coluna_estado):

    estados = {
        "AC": [-9.0479, -70.5265],
        "AL": [-9.6612, -36.6502],
        "AP": [1.3545, -51.9160],
        "AM": [-3.4653, -62.2159],
        "BA": [-12.9714, -38.5014],
        "CE": [-3.7172, -38.5433],
        "DF": [-15.7754, -47.7979],
        "ES": [-20.3155, -40.3128],
        "GO": [-16.6799, -49.2550],
        "MA": [-2.5387, -44.2827],
        "MT": [-12.6510, -56.5293],
        "MS": [-20.4486, -54.6295],
        "MG": [-18.5760, -44.0350],
        "PA": [-5.7945, -55.0974],
        "PB": [-7.1219, -36.7241],
        "PR": [-25.2521, -52.0215],
        "PE": [-8.0476, -34.8770],
        "PI": [-8.0476, -34.8770],
        "RJ": [-22.9068, -43.1729],
        "RN": [-5.7945, -36.2705],
        "RS": [-30.1097, -51.3174],
        "RO": [-11.5057, -63.5806],
        "RR": [2.7376, -62.0751],
        "SC": [-27.5954, -48.5480],
        "SP": [-23.5505, -46.6333],
        "SE": [-10.9472, -37.0731],
        "TO": [-10.9472, -37.0731]
    }

    if coluna_estado not in dataframe.columns:
        raise ValueError("A coluna de estados não existe no DataFrame.")

    dataframe = dataframe[dataframe[coluna_estado].isin(estados.keys())]
    
    mapa = folium.Map(location=[-14.235, -51.925], zoom_start=4)
    
    for index, row in dataframe.iterrows():
        estado = row[coluna_estado]
        coordenadas = estados[estado]
        folium.Marker(location=coordenadas, popup=estado).add_to(mapa)
    
    return mapa


