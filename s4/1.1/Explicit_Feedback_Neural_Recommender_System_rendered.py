import os.path as op
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
from keras import layers

ML_100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_100K_FILENAME = ML_100K_URL.rsplit('/', 1)[1]
ML_100K_FOLDER = 'ml-100k'

if not op.exists(ML_100K_FILENAME):
    print('Downloading %s to %s...' % (ML_100K_URL, ML_100K_FILENAME))
    urlretrieve(ML_100K_URL, ML_100K_FILENAME)

if not op.exists(ML_100K_FOLDER):
    print('Extracting %s to %s...' % (ML_100K_FILENAME, ML_100K_FOLDER))
    ZipFile(ML_100K_FILENAME).extractall('.')

# ### Ratings file
#
# Each line contains a rated movie:
# - a user
# - an item
# - a rating from 1 to 5 stars

# In[2]:


import pandas as pd

all_ratings = pd.read_csv(op.join(ML_100K_FOLDER, 'u.data'), sep='\t',
                          names=["user_id", "item_id", "rating", "timestamp"])
print(all_ratings.head())

# ### Item metadata file
# 
# The item metadata file contains metadata like the name of the movie or the date it was released

# In[3]:


names = ["name", "date", "genre", "url"]
names += ["f" + str(x) for x in range(19)]  # unused feature names

items = pd.read_csv(op.join(ML_100K_FOLDER, 'u.item'), sep='|', encoding='latin-1',
                    names=names)
# fix a missing value
items.fillna(value="01-Jan-1997", inplace=True)
print(items.head())

# ### Data preprocessing
# 
# To understand well the distribution of the data, the following statistics are computed:
# - the number of users
# - the number of items
# - the rating distribution

# In[4]:


print(all_ratings['rating'].describe())

# In[5]:


max_user_id = all_ratings['user_id'].max()
print(max_user_id)

# In[6]:


max_item_id = all_ratings['item_id'].max()
print(max_item_id)

# In[7]:


from sklearn.model_selection import train_test_split

ratings_train, ratings_test = train_test_split(
    all_ratings, test_size=0.2, random_state=0)

user_id_train = ratings_train['user_id']
item_id_train = ratings_train['item_id']
rating_train = ratings_train['rating']

user_id_test = ratings_test['user_id']
item_id_test = ratings_test['item_id']
rating_test = ratings_test['rating']

# # Explicit feedback: supervised ratings prediction
# 
# For each pair of (user, item) try to predict the rating the user would give to the item.
# 
# This is the classical setup for building recommender systems from offline data with explicit supervision signal. 

# ## Predictive ratings  as a regression problem
# 
# The following code implements the following architecture:
# 
# <img src="images/rec_archi_1.svg" style="width: 600px;" />

# In[8]:


from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.models import Model

# In[9]:


# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

embedding_size = 30
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

# y = merge([user_vecs, item_vecs], mode='dot', output_shape=(1,))
y = layers.dot([user_vecs, item_vecs], axes=1)

model = Model(inputs=[user_id_input, item_id_input], outputs=[y])
model.compile(optimizer='adam', loss='mae')

# In[10]:


# Useful for debugging the output shape of model
initial_train_preds = model.predict([user_id_train, item_id_train]).squeeze()
print(initial_train_preds.shape)

# ### Model error
# 
# Using `initial_train_preds`, compute the model errors:
# - mean absolute error
# - mean squared error
# 
# Converting a pandas Series to numpy array is usually implicit, but you may use `rating_train.values` to do so explicitely. Be sure to monitor the shapes of each object you deal with by using `object.shape`.

# In[11]:


# %load solutions/compute_errors.py
squared_differences = np.square(initial_train_preds - rating_train.values)
absolute_differences = np.abs(initial_train_preds - rating_train.values)

print("Random init MSE: %0.3f" % np.mean(squared_differences))
print("Random init MAE: %0.3f" % np.mean(absolute_differences))

# You may also use sklearn metrics to do so with less numpy engineering 
# from sklearn.metrics import mean_squared_error, mean_absolute_error
#
# print("Random init MSE: %0.3f" % mean_squared_error(initial_train_preds, rating_train))
# print("Random init MAE: %0.3f" % mean_absolute_error(initial_train_preds, rating_train))


# ### Monitoring runs
# 
# Keras enables to monitor various variables during training. 
# 
# `history.history` returned by the `model.fit` function is a dictionary
# containing the `'loss'` and validation loss `'val_loss'` after each epoch

# In[12]:

# Training the model
history = model.fit([user_id_train, item_id_train], rating_train,
                    batch_size=64, epochs=20, validation_split=0.1,
                    shuffle=True, verbose=2)

# In[13]:


# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='validation')
# plt.ylim(0, 2)
# plt.legend(loc='best')
# plt.title('Loss');

# Now that the model is trained, the model MSE and MAE look nicer:

# In[14]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

test_preds = model.predict([user_id_test, item_id_test]).squeeze()
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))

# In[15]:


train_preds = model.predict([user_id_train, item_id_train]).squeeze()
print("Final train MSE: %0.3f" % mean_squared_error(train_preds, rating_train))
print("Final train MAE: %0.3f" % mean_absolute_error(train_preds, rating_train))

# ## A Deep recommender model
# 
# Using a similar framework as previously, the following deep model described in the course was built (with only two fully connected)
# 
# <img src="images/rec_archi_2.svg" style="width: 600px;" />
# 
# 
# ### Exercise
# 
# - The following code has **4 errors** that prevent it from working correctly. **Correct them and explain** why they are critical.

# In[16]:

# %load solutions/deep_explicit_feedback_recsys.py
# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

embedding_size = 30
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = layers.concatenate([user_vecs, item_vecs])
## Error 1: Dropout was too high, preventing any training
input_vecs = Dropout(0.5)(input_vecs)

x = Dense(64, activation='relu')(input_vecs)

## Error 2: output dimension was 2 where we predict only 1-d rating
## Error 3: tanh activation squashes the outputs between -1 and 1
## when we want to predict values between 1 and 5
y = Dense(1)(x)

model = Model(inputs=[user_id_input, item_id_input], outputs=[y])
## Error 4: A binary crossentropy loss is only useful for binary
## classification, while we are in regression (use mse or mae)
model.compile(optimizer='adam', loss='mae')

initial_train_preds = model.predict([user_id_train, item_id_train]).squeeze()

# In[18]:
history = model.fit([user_id_train, item_id_train], rating_train,
                    batch_size=64, epochs=20, validation_split=0.1,
                    shuffle=True, verbose=2)

# In[20]:


test_preds = model.predict([user_id_test, item_id_test]).squeeze()
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))

# In[21]:


train_preds = model.predict([user_id_train, item_id_train]).squeeze()
print("Final train MSE: %0.3f" % mean_squared_error(train_preds, rating_train))
print("Final train MAE: %0.3f" % mean_absolute_error(train_preds, rating_train))

# ### Home assignment:
#  - Add another layer, compare train/test error
#  - What do you notice? 
#  - Try adding more dropout and modifying layer sizes: should you increase
#    or decrease the number of parameters

# ### Model Embeddings
# 
# - It is possible to retrieve the embeddings by simply using the Keras function `model.get_weights` which returns all the model learnable parameters.
# - The weights are returned the same order as they were build in the model
# - What is the total number of parameters?

# In[22]:


# weights and shape
weights = model.get_weights()
print([w.shape for w in weights])

# In[23]:


# Solution: 
# model.summary()


# In[24]:


user_embeddings = weights[0]
item_embeddings = weights[1]
print("First item name from metadata:", items["name"][1])
print("Embedding vector for the first item:")
print(item_embeddings[1])
print("shape:", item_embeddings[1].shape)

# ### Finding most similar items
# Finding k most similar items to a point in embedding space
# 
# - Write in numpy a function to compute the cosine similarity between two points in embedding space
# - Write a function which computes the euclidean distance between a point in embedding space and all other points
# - Write a most similar function, which returns the k item names with lowest euclidean distance
# - Try with a movie index, such as 181 (Return of the Jedi). What do you observe? Don't expect miracles on such a small training set.
# 
# Notes:
# - you may use `np.linalg.norm` to compute the norm of vector, and you may specify the `axis=`
# - the numpy function `np.argsort(...)` enables to compute the sorted indices of a vector
# - `items["name"][idxs]` returns the names of the items indexed by array idxs

# In[25]:


EPSILON = 1e-07


def cosine(x, y):
    # TODO: modify function
    return 0.


# Computes euclidean distances between x and all item embeddings
def euclidean_distances(x):
    # TODO: modify function
    return 0.


# Computes top_n most similar items to an idx
def most_similar(idx, top_n=10):
    # TODO: modify function
    idxs = np.array([1, 2, 3])
    return items["name"][idxs]


print(most_similar(181))

# In[26]:


# %load solutions/similarity.py
EPSILON = 1e-07


def cosine(x, y):
    dot_pdt = np.dot(x, y.T)
    norms = np.linalg.norm(x) * np.linalg.norm(y)
    return dot_pdt / (norms + EPSILON)


# Computes cosine similarities between x and all item embeddings
def cosine_similarities(x):
    dot_pdts = np.dot(item_embeddings, x)
    norms = np.linalg.norm(x) * np.linalg.norm(item_embeddings, axis=1)
    return dot_pdts / (norms + EPSILON)


# Computes euclidean distances between x and all item embeddings
def euclidean_distances(x):
    return np.linalg.norm(item_embeddings - x, axis=1)


# Computes top_n most similar items to an idx,
def most_similar(idx, top_n=10, mode='euclidean'):
    sorted_indexes = 0
    if mode == 'euclidean':
        dists = euclidean_distances(item_embeddings[idx])
        sorted_indexes = np.argsort(dists)
        idxs = sorted_indexes[0:top_n]
        return list(zip(items["name"][idxs], dists[idxs]))
    else:
        sims = cosine_similarities(item_embeddings[idx])
        # [::-1] makes it possible to reverse the order of a numpy
        # array, this is required because most similar items have
        # a larger cosine similarity value
        sorted_indexes = np.argsort(sims)[::-1]
        idxs = sorted_indexes[0:top_n]
        return list(zip(items["name"][idxs], sims[idxs]))


# sanity checks:
print("cosine of item 1 and item 1: "
      + str(cosine(item_embeddings[1], item_embeddings[1])))
euc_dists = euclidean_distances(item_embeddings[1])
print(euc_dists.shape)
print(euc_dists[1:5])
print()

# Test on movie 181: Return of the Jedi
print("Items closest to 'Return of the Jedi':")
for title, dist in most_similar(181, mode="euclidean"):
    print(title, dist)

# We observe that the embedding is poor at representing similarities
# between movies, as most distance/similarities are very small/big 
# One may notice a few clusters though
# it's interesting to plot the following distributions
# plt.hist(euc_dists)

# The reason for that is that the number of ratings is low and the embedding
# does not automatically capture semantic relationships in that context. 
# Better representations arise with higher number of ratings, and less overfitting models


# ### Visualizing embeddings using TSNE
# 
# - we use scikit learn to visualize items embeddings
# - Try different perplexities, and visualize user embeddings as well
# - What can you conclude ?

# In[27]:


from sklearn.manifold import TSNE

item_tsne = TSNE(perplexity=30).fit_transform(item_embeddings)

# In[28]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.scatter(item_tsne[:, 0], item_tsne[:, 1])
plt.xticks(())
plt.yticks(())
plt.show()

# ## Using item metadata in the model
# 
# Using a similar framework as previously, we will build another deep model that can also leverage additional metadata. The resulting system is therefore an **Hybrid Recommender System** that does both **Collaborative Filtering** and **Content-based recommendations**.
# 
# <img src="images/rec_archi_3.svg" style="width: 600px;" />
# 

# In[29]:


# transform the date (string) into an int representing the release year
parsed_dates = [int(film_date[-4:])
                for film_date in items["date"].tolist()]

items['parsed_date'] = pd.Series(parsed_dates, index=items.index)
max_date = max(items['parsed_date'])
min_date = min(items['parsed_date'])

from sklearn.preprocessing import scale

items['scaled_date'] = scale(items['parsed_date'].astype('float64'))
item_meta_train = items["scaled_date"][item_id_train]
item_meta_test = items["scaled_date"][item_id_test]

len(item_meta_train), len(item_meta_test)

# In[30]:


items["scaled_date"].describe()

# In[31]:


# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')
meta_input = Input(shape=[1], name='meta_item')

embedding_size = 32
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = layers.concatenate([user_vecs, item_vecs, meta_input])

x = Dense(64, activation='relu')(input_vecs)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
y = Dense(1)(x)

model = Model(inputs=[user_id_input, item_id_input, meta_input], outputs=y)
model.compile(optimizer='adam', loss='mae')

initial_train_preds = model.predict([user_id_train, item_id_train, item_meta_train]).squeeze()

# In[32]:
history = model.fit([user_id_train, item_id_train, item_meta_train], rating_train,
                    batch_size=64, epochs=20, validation_split=0.1,
                    shuffle=True, verbose=2)

# In[33]:


test_preds = model.predict([user_id_test, item_id_test, item_meta_test]).squeeze()
print("Final test Loss: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test Loss: %0.3f" % mean_absolute_error(test_preds, rating_test))


# ### A recommendation function for a given user
# 
# Once the model is trained, the system can be used to recommend a few items for a user, that he/she hasn't already seen:
# - we use the `model.predict` to compute the ratings a user would have given to all items
# - we build a reco function that sorts these items and exclude those the user has already seen

# In[34]:


def recommend(user_id, top_n=10):
    item_ids = range(1, max_item_id)
    seen_movies = list(all_ratings[all_ratings["user_id"] == user_id]["item_id"])
    item_ids = list(filter(lambda x: x not in seen_movies, item_ids))

    print("user " + str(user_id) + " has seen " + str(len(seen_movies)) + " movies. " +
          "Computing ratings for " + str(len(item_ids)) + " other movies")

    item_ids = np.array(item_ids)
    user = np.zeros_like(item_ids)
    user[:] = user_id
    items_meta = items["scaled_date"][item_ids].values

    rating_preds = model.predict([user, item_ids, items_meta])

    item_ids = np.argsort(rating_preds[:, 0])[::-1].tolist()
    rec_items = item_ids[:top_n]
    return [(items["name"][movie], rating_preds[movie][0]) for movie in rec_items]


# In[35]:


print(recommend(3))

# ### Home assignment: Predicting ratings as a classification problem
# 
# In this dataset, the ratings all belong to a finite set of possible values:

# In[36]:


import numpy as np

np.unique(rating_train)

# Maybe we can help the model by forcing it to predict those values by treating the problem as a multiclassification problem. The only required changes are:
# 
# - setting the final layer to output class membership probabities using a softmax activation with 5 outputs;
# - optimize the categorical cross-entropy classification loss instead of a regression loss such as MSE or MAE.

# In[37]:


# %load solutions/classification.py
# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

embedding_size = 16
dense_size = 128
dropout_embedding = 0.5
dropout_hidden = 0.2

user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = layers.concatenate([user_vecs, item_vecs])
input_vecs = Dropout(dropout_embedding)(input_vecs)

x = Dense(dense_size, activation='relu')(input_vecs)
x = Dropout(dropout_hidden)(x)
x = Dense(dense_size, activation='relu')(x)
y = Dense(5, activation='softmax')(x)

model = Model(inputs=[user_id_input, item_id_input], outputs=y)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

initial_train_preds = model.predict([user_id_train, item_id_train]).squeeze().argmax(axis=1) + 1
print("Random init MSE: %0.3f" % mean_squared_error(initial_train_preds, rating_train))
print("Random init MAE: %0.3f" % mean_absolute_error(initial_train_preds, rating_train))

history = model.fit([user_id_train, item_id_train], rating_train - 1,
                    batch_size=64, epochs=20, validation_split=0.1,
                    shuffle=True, verbose=2)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0, 2)
plt.legend(loc='best')
plt.title('loss')

test_preds = model.predict([user_id_test, item_id_test]).squeeze().argmax(axis=1) + 1
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))
