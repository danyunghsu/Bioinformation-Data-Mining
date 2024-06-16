import re
from tqdm import tqdm
import keras as K
import numpy as np
from sklearn.metrics import roc_curve,auc,roc_auc_score,confusion_matrix
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Add, Activation, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.metrics import AUC
import gc
import lightgbm as lgb
import xgboost as xgb

def seq2onehot(seq):
    dic = {'a': [1, 0, 0,0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1]}
    seq = seq.strip()
    onehot = []
    for nt in seq:
        if nt not in dic.keys():
            nt = '*'
        temp = dic[nt]
        onehot+=temp
    return onehot

def onehot(seq):
    #results = list(map(seq2onehot, seq))
    results = list(seq2onehot(seq))
    np_results = np.array(results)
    np_results = np_results.flatten()
    result=np_results.tolist()
    return(result)

def Prepare_DNN(input_shape):
    input_img = Input(shape=(input_shape,))
    x = Dense(5, activation='relu')(input_img)
    decoded = Dense(1,activation='sigmoid')(x)
    Adam = K.optimizers.Adam(lr=1e-6)
    classifier = Model(input_img, decoded)
    classifier.compile(optimizer=Adam, loss='binary_crossentropy',metrics=[AUC(name='auc')])

    classifier.summary()
    return(classifier)

def get_data(folder_path):
    file_names = os.listdir(folder_path)
    donor_seqs =[]
    not_seqs = []
    for file in tqdm(file_names):
        file_path = os.path.join(folder_path,file)
        with open(file_path,'r') as f:
            lines = f.readlines()
            donor_site = re.findall('(\d+)(?=,)',lines[1])
            seq = ''.join(lines[2:]).replace('\n','').lower()
            for pos in donor_site:
                donor_seqs.append(seq[int(pos)-4:int(pos)+5])

            for pos in range(len(seq) - 8):
                not_seq = seq[pos:pos+9]

            if not_seq not in donor_seqs and set(not_seq) == {'a','c','t','g'}:
                not_seqs.append(not_seq)

    return  donor_seqs,not_seqs

start=time.time()
train_path='/content/drive/MyDrive/data/train'
test_path = '/content/drive/MyDrive/data/test'
#train_donor,train_not = get_data(train_path)
test_donor,test_not = get_data(test_path)
train_pos = []
train_neg = []
test_pos = []
test_neg = []
# for i in range(len(train_donor)):
#     train_pos.append(onehot(train_donor[i]))
# for i in range(len(train_not)):
#     train_neg.append(onehot(train_not[i]))
for i in range(len(test_donor)):
    test_pos.append(onehot(test_donor[i]))
for i in range(len(test_not)):
    test_neg.append(onehot(test_not[i]))

# train_pos_ar = np.array(train_pos)
# train_neg_ar = np.array(train_neg)
test_pos_ar = np.array(test_pos)
test_neg_ar = np.array(test_neg)
#print(train_pos_ar.shape,train_neg_ar.shape,test_pos_ar.shape,test_neg_ar.shape)

'''
tot_label = []
tot_score = []
for i in range(10):
    pos_batch_num = int(train_pos_ar.shape[0]/10)
    neg_batch_num = int(train_neg_ar.shape[0]/10)

    vali_pos = train_pos_ar[i*pos_batch_num:(i+1)*pos_batch_num,:]
    vali_neg = train_neg_ar[i*neg_batch_num:(i+1)*neg_batch_num,:]
    train_pos = np.vstack((train_pos_ar[0:i*pos_batch_num,:],train_pos_ar[(i+1)*pos_batch_num:,:]))
    train_neg = np.vstack((train_neg_ar[0:i*neg_batch_num,:],train_neg_ar[(i+1)*neg_batch_num:,:]))

    train_x = np.vstack((train_pos,train_neg))
    train_y = np.array([1]*train_pos.shape[0]+[0]*train_neg.shape[0])
    vali_x = np.vstack((vali_pos,vali_neg))
    vali_y = np.array([1]*vali_pos.shape[0]+[0]*vali_neg.shape[0])
    print(train_x.shape,vali_x.shape)

    train_data = lgb.Dataset(data=train_x,label=train_y)
    test_data = lgb.Dataset(data=vali_x,label=vali_y)
    num_round = 10
    params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'learning_rate': 1e-5,
                'metric': 'auc',
                'min_child_weight': 1e-3,
                'num_leaves': 31,
                'max_depth': -1,
                'reg_lambda': 0,
                'reg_alpha': 0,
                'feature_fraction': 1,
                'bagging_fraction': 1,
                'bagging_freq': 0,
                'seed': 2020,
                'nthread': 8,
                'silent': True,
                'verbose': -1,
            }
    bst = lgb.train(params,train_set=train_data,valid_sets=test_data,num_boost_round=10, callbacks=[lgb.log_evaluation(period=10, show_stdv=True)])
    bst.save_model("/content/drive/MyDrive/gbm/lgb_{0}.txt".format(i))
    predict_y = bst.predict(vali_x)
    tot_label += vali_y.tolist()
    tot_score += predict_y.tolist()

file_out = open("/content/drive/MyDrive/gbm/gbm_10fold.txt",'w')
for i in range(len(tot_label)):
    file_out.write(str(tot_label[i])+"\t"+str(tot_score[i])+"\n")
AUC_score = roc_auc_score(tot_label, tot_score)
file_out.write("AUC_score:"+str(AUC_score))
file_out.close()
print('10 fold auc', AUC_score)
'''
test_x = np.vstack((test_pos_ar,test_neg_ar))
test_y = np.array([1]*test_pos_ar.shape[0]+[0]*test_neg_ar.shape[0])
for i in range(1):
  test_model = lgb.Booster(model_file="/content/drive/MyDrive/gbm/lgb_1.txt")#可以更换哪一个模型
  predict_test = test_model.predict(test_x)
  test_auc = roc_auc_score(test_y,predict_test)
  print('{0} model on test'.format(i),test_auc)
  #gc.collect()

y_pred_binary = [1 if pred > 0.5 else 0 for pred in predict_test]
cm = confusion_matrix(test_y, y_pred_binary)
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print("Sensitivity (Sn):", sensitivity)
print("Specificity (Sp):", specificity)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('LGB Confusion Matrix')
plt.savefig("/content/drive/MyDrive/gbm/cm.png")
plt.show()