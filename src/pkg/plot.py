import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import folium
import shap

from sklearn import decomposition

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

def pca_2d(X, y=None, savename=False):
    '''
    receives X features
    return a scatter plot of 2 components pca
    and color it with third component
    '''
    fig = plt.figure(figsize=(10,10))

    pca = decomposition.PCA(n_components=3)
    view = pca.fit_transform(X)

    if y is not None:
        plt.title(f'PCA colored by {y.name}')
        plt.scatter(view[:,0], view[:,1], c=y, cmap='viridis')
        plt.colorbar(label=y.name)
    else:
        plt.scatter(view[:,0], view[:,1], c=view[:,2])
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')

    if savename :
        save(savename, fig.savefig)

    plt.show()

def pca_3d(X, y=None, multiple_graph=False, savename=False):
    '''
    Plot a 3-component PCA.
    If multiple_graph: plot a 2x2 grid with a 3D PCA in position 1x1, and three 2D combinations in other positions.
    If not multiple_graph: just plot a 3D PCA.
    '''

    pca = decomposition.PCA(n_components=4)
    view = pca.fit_transform(X)

    if not multiple_graph:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        if y is not None:
            scatter = ax.scatter(view[:,0], view[:,1], view[:,2], c=y, cmap='viridis')
            fig.colorbar(scatter, label=y.name)
        else:
            ax.scatter(view[:,0], view[:,1], view[:,2],  c=view[:,3], cmap='viridis')

        ax.set_xlabel('Comp 1')
        ax.set_ylabel('Comp 2')
        ax.set_zlabel('Comp 3')


    else:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()

        combinations = [(0,0), (0, 1), (0, 2), (1, 2)]

        for i, (x_idx, y_idx) in enumerate(combinations):
            if i == 0:
                ax = fig.add_subplot(2, 2, 1, projection='3d')
                if y is not None:
                    scatter = ax.scatter(view[:,0], view[:,1], view[:,2], c=y, cmap='viridis')
                else:
                    ax.scatter(view[:,0], view[:,1], view[:,2], c=view[:,3], cmap='viridis')
                ax.set_xlabel('Comp 1')
                ax.set_ylabel('Comp 2')
                ax.set_zlabel('Comp 3')
            else:
                ax = axs[i]
                if y is not None:
                    scatter = ax.scatter(view[:, x_idx], view[:, y_idx], c=y, cmap='viridis')
                else:
                    ax.scatter(view[:, x_idx], view[:, y_idx], c=view[:,3], cmap='viridis')
                ax.set_xlabel(f'Comp {x_idx+1}')
                ax.set_ylabel(f'Comp {y_idx+1}')

        if y is not None:
            fig.colorbar(scatter, ax=axs.ravel(), label=y.name)
            fig.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.95, wspace=0.3, hspace=0.2)
        else:
            fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.2, hspace=0.2)
        
    if y is not None:
        fig.suptitle(f'PCA colored by {y.name}', fontsize=16)
    else:
        fig.suptitle('PCA colored by next component', fontsize=16)
        
    if savename:
        save(savename, plt.savefig)

    plt.show()

def shapping_rf(rf, sample, savename=None):

    fig = plt.figure(figsize=(10,10))

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample)

    if savename :
        save(savename, fig.savefig)


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
        save(savename, plt.savefig)
    
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

