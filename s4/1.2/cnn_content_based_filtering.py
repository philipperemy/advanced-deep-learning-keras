import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage

np.set_printoptions(threshold=np.nan)
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('ml-latest-small/ratings.csv', sep=',')
df_id = pd.read_csv('ml-latest-small/links.csv', sep=',')
df_movie_names = pd.read_csv('ml-latest-small/movies.csv', sep=',')
df = pd.merge(pd.merge(df, df_id, on='movieId'), df_movie_names, on='movieId')

print(df.head())

data_file = 'imdb_id_to_image_dict.data'
if not os.path.exists(data_file):
    imdb_id_to_image_dict = dict()
    for poster_file in glob('posters/*.jpg'):  # debug here
        print('Loading img at {}'.format(poster_file))
        img = kimage.load_img(poster_file, target_size=(224, 224))
        img = preprocess_input(np.expand_dims(kimage.img_to_array(img), axis=0))
        imdb_id = poster_file.split('/')[-1].split('.')[0]
        imdb_id_to_image_dict[imdb_id] = img
    pickle.dump(file=open(data_file, 'wb'), obj=imdb_id_to_image_dict)
else:
    imdb_id_to_image_dict = pickle.load(file=open(data_file, 'rb'))

model = VGG16(include_top=False, weights='imagenet')

imdb_id_to_title = {}
for row in df.itertuples():
    if row.imdbId not in imdb_id_to_title:
        print(row.imdbId, row.title)
    imdb_id_to_title[row.imdbId] = row.title

matrix_id_to_imdb_id = dict()
imdb_id_to_matrix_id = dict()
for i, (imdb_id, img) in enumerate(imdb_id_to_image_dict.items()):  # imdb ids.
    matrix_id_to_imdb_id[i] = imdb_id
    imdb_id_to_matrix_id[imdb_id] = i

num_movies = len(imdb_id_to_image_dict.keys())
print('Number of movies = {}'.format(num_movies))
predictions_file = 'matrix_res.npz'
if not os.path.exists(predictions_file):
    matrix_res = np.zeros([num_movies, 25088])
    for i, (imdb_id, img) in enumerate(imdb_id_to_image_dict.items()):  # imdb ids.
        print('Predicting for imdb_id = {}'.format(imdb_id))
        matrix_res[i, :] = model.predict(img).ravel()
    np.savez_compressed(file=predictions_file, matrix_res=matrix_res)
else:
    matrix_res = np.load(predictions_file)['matrix_res']

similarity_deep = matrix_res.dot(matrix_res.T)
norms = np.array([np.sqrt(np.diagonal(similarity_deep))])
similarity_deep = similarity_deep / (norms * norms.T)
print(similarity_deep.shape)

target_imdb_id = '114709'  # toy story.
target_id = imdb_id_to_matrix_id[target_imdb_id]
print('target_id =', target_id)

closest_movies_ids = np.argsort(similarity_deep[target_id, :])[::-1][0:20]
closest_movies_imdb_ids = [matrix_id_to_imdb_id[matrix_id] for matrix_id in closest_movies_ids]
closest_movies_titles = [imdb_id_to_title[int(imdb_id)] for imdb_id in closest_movies_imdb_ids]
print(closest_movies_titles)
