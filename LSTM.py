#coding=utf-8
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, BatchNormalization
import collections
import keras.backend as K
import math
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping



def mean_pred(y_ture, y_pred):
    return K.mean(abs(y_ture - y_pred))


def scheduler(epoch):
    # 每隔8个epoch，学习率减小为原来的1/2
    if epoch % 8 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)


NUM_LSTM_UNITS = 128
BATCH_SIZE = 16
steps = 5  # time slices
features = 11*81  # stations*lines
model = Sequential()
model.add(LSTM(units=NUM_LSTM_UNITS, input_shape=(steps, features), return_sequences=True))
model.add(LSTM(units=NUM_LSTM_UNITS))
model.add(BatchNormalization())
model.add(RepeatVector(81))
model.add(Dense(11))
model.compile(loss= mean_pred, optimizer='adam', metrics=[mean_pred])
model.summary()


t = np.load('normal_tensor1_2_4_part_0_transfer_for_rnn.npz')
t0 = np.load('normal_tensor1_2_4_part_1_transfer_for_rnn.npz')
t2 = np.load('normal_tensor1_2_4_part_2_transfer_for_rnn.npz')
t3 = np.load('normal_tensor1_2_4_part_3_transfer_for_rnn.npz')
t4 = np.load('normal_tensor1_2_4_part_4_transfer_for_rnn.npz')
t6 = np.load('normal_tensor1_2_4_part_6_transfer_for_rnn.npz')
#-----task3 -----
t1 = np.concatenate((t['t4'], t0['t4'], t2['t4'], t3['t4'], t4['t4'], t6['t4']), axis=0)
t1_target = np.concatenate((t['t4_target'], t0['t4_target'], t2['t4_target'], t3['t4_target'], t4['t4_target'], t6['t4_target']), axis=0)

# checkpoint
filepath = "log/LSTM_task1/{epoch:02d}-{loss:.8f}.hdf5"
# 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
reduce_lr = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0,  mode='min')
callbacks_list = [checkpoint, reduce_lr, early_stopping]
K.set_value(model.optimizer.lr, 0.001)
model.fit(t1, t1_target, batch_size=BATCH_SIZE, epochs=50, callbacks=callbacks_list)
