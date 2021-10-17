import tensorflow as tf
from binary_game import binary_game as bgc

import numpy as np
import pandas as pd

n=5 #number of inputs is 2^(n-1) = 16
k=2 #number of weights = (n+1)*k = 18

#scores = [1,3,5,7]
scores = np.random.randint(10, size = 2**(n-1))

BATCH_SIZE = 200

model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(k, input_shape = (n,), activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(1, activation="relu")]
)

bg = bgc(n, scores = scores, model=model, alpha = 1)

for i in range(10):
    batch = [None]*BATCH_SIZE
    for i in range(BATCH_SIZE):
        batch[i] = bg.play(True)j

    results = np.concatenate(batch)

    #reduce loss by taking average first
    df = pd.DataFrame(results)
    df_mean = df.groupby(list(df.columns)[:-1]).mean().reset_index()
    df = df.drop(columns = list(df.columns)[-1])
    df = df.merge(df_mean)

    results = df.to_numpy()

    model.compile(optimizer='sgd',
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    model.fit(results[:, :-1], results[:,-1], epochs=5)

states, scores = bg.dfs()
df = pd.DataFrame(np.array(states))
df['predict'] = model(df.to_numpy())
df = df.merge(df_mean, how="left")
df['optimal_scores'] = scores
print(df)
print(scores)

print(model.layers[0].get_weights())
print(model.layers[1].get_weights())