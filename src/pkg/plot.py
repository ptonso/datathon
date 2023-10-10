import os

import matplotlib.pyplot as plt
import seaborn as sns

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