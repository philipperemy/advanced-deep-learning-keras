import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

dataset = datasets.load_diabetes()

print(dataset.data.shape)
print(dataset.target.shape)

model = Ridge()
model.fit(dataset.data, dataset.target)
print('Score with default parameters = ', model.score(dataset.data, dataset.target))

alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])

grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(dataset.data, dataset.target)

print('Score with Grid Search parameters =', grid.best_score_, 'best alpha =', grid.best_estimator_.alpha)

param_grid = {'alpha': sp_rand()}

rand_grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
rand_grid_search.fit(dataset.data, dataset.target)

print('Score with Random Search parameters =', rand_grid_search.best_score_, 'best alpha =',
      rand_grid_search.best_estimator_.alpha)
