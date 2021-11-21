import os
import pickle
import random
import copy

import librosa
import mirdata
import openl3
import numpy as np
import tensorflow as tf
import sklearn
import pandas as pd
from IPython.display import display, Audio

from tqdm import tqdm

MIRDATA_MDSB_PATH = './mirdatasets/medley_solos_db'
SR = 48000
NUM_CLASSES = 8
MODEL_PATH = './inst_rec_tutorial.mdl'


def get_instrument_df(subset='training'):
    msdb = mirdata.initialize('medley_solos_db', data_home=MIRDATA_MDSB_PATH)
    
    inst_df = pd.DataFrame.from_records([dict(instrument_id=msdb.track(track_id).instrument_id,
                                              instrument=str(msdb.track(track_id).instrument_id) + ' ' +
                                              msdb.track(track_id).instrument) for track_id in msdb.track_ids
                                         if msdb.track(track_id).subset==subset])
    
    return inst_df


def get_class_weights(inst_df):
    class_weights = (1 / inst_df.groupby('instrument_id').count()) * (inst_df.shape[0] / NUM_CLASSES)
    return class_weights.to_dict()['instrument']


def msdb_generator(subset='training', limit=-1, data_home=MIRDATA_MDSB_PATH):
    msdb = mirdata.initialize('medley_solos_db', data_home=MIRDATA_MDSB_PATH)
    track_ids = [t_id for t_id in msdb.track_ids if msdb.track(t_id).subset==subset]
    track_ids = track_ids[:limit]
    
    random.shuffle(track_ids)
    
    for track_id in track_ids:
        track = msdb.track(track_id)
        audio, sr = librosa.load(track.audio_path, sr=SR)
        for x in librosa.util.frame(audio, frame_length=SR, hop_length=SR//2).T:
            yield x[np.newaxis,:].astype(np.float32), track.instrument_id
            
            
def build_model(ol3_trainable=False):
    ol3 = openl3.models.load_audio_embedding_model(input_repr='mel256', content_type='music',
                                                   embedding_size=512)
    ol3.trainable = ol3_trainable
    
    x_a = tf.keras.layers.Input(shape=(1,SR))
    y_a = ol3(x_a)
    
    y_a = tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax")(y_a)
    model = tf.keras.Model(inputs=x_a, outputs=y_a)
    
    return model


def train_model(config, model=None, model_chkpt_path=MODEL_PATH):
    msdb_training_dataset = tf.data.Dataset.from_generator(lambda: msdb_generator(subset='training',
                                                                                  limit=config['training_set_limit']),
                                                           output_types=(tf.float32, tf.int32),
                                                           output_shapes=((1,SR), ()))
    
    msdb_training_dataset = msdb_training_dataset.shuffle(buffer_size=config['buffer_size']).repeat().batch(config['batch_size'])
    
    msdb_validation_dataset = tf.data.Dataset.from_generator(lambda: msdb_generator(subset='validation',
                                                                                    limit=config['validation_set_limit']),
                                                             output_types=(tf.float32, tf.int32),
                                                             output_shapes=((1,SR), ()))

    msdb_validation_dataset = msdb_validation_dataset.batch(config['batch_size'])
    
    inst_df = get_instrument_df()
    class_weights = get_class_weights(inst_df)
    
    if model is None:
        model = build_model()
        
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
        
    callbacks = [tf.keras.callbacks.ModelCheckpoint(
                     filepath=model_chkpt_path,
                     save_weights_only=True,
                     monitor='val_accuracy',
                     mode='max',
                     save_best_only=True),]
    
    history = model.fit(msdb_training_dataset, 
                        batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        steps_per_epoch=config['steps_per_epoch'],
                        validation_data=msdb_validation_dataset,
                        callbacks=callbacks,
                        class_weight=class_weights)
    
    return model, history
        

def eval_on_test(model):
    msdb = mirdata.initialize('medley_solos_db', data_home=MIRDATA_MDSB_PATH)
    track_ids = [t_id for t_id in msdb.track_ids if msdb.track(t_id).subset=='test']

    y_pred = []
    y_true = []
    for track_id in tqdm(track_ids):
        track = msdb.track(track_id)
        audio, sr = librosa.load(track.audio_path, sr=SR)
        x = librosa.util.frame(audio, frame_length=SR, hop_length=SR // 2).T
        l = model.predict(x[:,np.newaxis,:])
        _y_pred = np.argmax(np.mean(l, axis=0))
        y_pred.append(_y_pred)
        y_true.append(track.instrument_id)

    print(sklearn.metrics.classification_report(y_true, y_pred))
    
    matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    print(np.mean(matrix.diagonal()/matrix.sum(axis=1)))
        
    return y_true, y_pred, confusion_matrix