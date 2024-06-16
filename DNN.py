from sklearn.svm import SVC
import re
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
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
    Adam = K.optimizers.Adam(lr=1e-4)
    classifier = Model(input_img, decoded)
    classifier.compile(optimizer=Adam, loss='binary_crossentropy',metrics=[AUC(name='auc')])

    classifier.summary()
    return(classifier)

def Train_DNN(train_x,train_y,test_x,test_y,model_path,input_shape):
    DNN_model = Prepare_DNN(input_shape)
    #DNN_model = K.models.load_model('./MixModel/MixModel_0/DNN_Best.model')
    earlyStop = EarlyStopping(monitor="val_auc",min_delta=0,patience=2,mode="max")
    history1  = DNN_model.fit(train_x, train_y, epochs=20, batch_size=8,validation_data=(test_x,test_y),class_weight={0:1,1:1},shuffle=True,callbacks=[earlyStop,ModelCheckpoint(model_path+"/DNN_Best.model", monitor="val_auc", mode="max", save_best_only=True)])
    lossy = history1.history['loss']
    np_lossy =np.array(lossy).reshape((1,len(lossy)))
    np_out = np.concatenate([np_lossy],axis=0)
    np.savetxt(model_path+'/loss.txt',np_out)

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
                
    # not_seqs_array = np.array(not_seqs)
    # donor_seqs_array = np.array(donor_seqs)            
    return  donor_seqs,not_seqs

start=time.time()

train_path='D:/data_mining/data/train'
test_path = 'D:/data_mining/data/test'
train_donor,train_not = get_data(train_path)
test_donor,test_not = get_data(test_path)
train_pos = []
train_neg = []
test_pos = []
test_neg = []
for i in range(len(train_donor)):
    train_pos.append(onehot(train_donor[i]))
for i in range(len(train_not)):
    train_neg.append(onehot(train_not[i]))
for i in range(len(test_donor)):
    test_pos.append(onehot(test_donor[i]))
for i in range(len(test_not)):
    test_neg.append(onehot(test_not[i]))

train_pos_ar = np.array(train_pos)
train_neg_ar = np.array(train_neg)
test_pos_ar = np.array(test_pos)
test_neg_ar = np.array(test_neg)

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

    Train_DNN(train_x,train_y,vali_x,vali_y,"./MixModel/MixModel_{0}".format(i),train_x.shape[1])
    model_MixModel =  K.models.load_model("./MixModel/MixModel_{0}/DNN_Best.model".format(i))
    predict_y = model_MixModel.predict(vali_x)
    tot_label += vali_y.tolist()
    tot_score += predict_y.tolist()

file_out = open("./MixModel/MixModel_result.txt",'w')
for i in range(len(tot_label)):
    file_out.write(str(tot_label[i])+"\t"+str(tot_score[i][0])+"\n")
AUC_score = roc_auc_score(tot_label, tot_score)
file_out.write("AUC_score:"+str(AUC_score))
file_out.close()
print('auc', AUC_score)

test_x = np.vstack((test_pos_ar,test_neg_ar))
test_y = np.array([1]*test_pos_ar.shape[0]+[0]*test_neg_ar.shape[0])
test_model = K.models.load_model("./Mix_Model_Data/AutoEncoder/AutoEncoder/DNN_Best.model")#可以更换哪一个模型
predict_test = test_model.predict(test_x)
test_auc = roc_auc_score(test_y,predict_y)
print(test_auc)

'''
svm = SVC(gamma=1,C=1,probability=True)
svm.fit(x_train,y_train)
y_pred = svm.predict_proba(x_test)[:,1]
test_label = y_test.tolist()
test_score = y_pred.tolist()
y_pred_binary = [1 if pred > 0.5 else 0 for pred in test_score]
cm = confusion_matrix(test_label, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

file_out = open('D:\data_mining\SVM\SVM_result.txt','w')
for i in range(len(test_label)):
    file_out.write(str(test_label[i])+'\t'+str(test_score[i])+'\n')

auc = roc_auc_score(test_label,test_score)
fpr,tpr,thres = roc_curve(test_label,test_score,)
print('AUC ', auc)
fig,ax = plt.subplots(figsize=(10,8))
ax.plot(fpr,tpr,linewidth=2,label='SVM (AUC={})'.format(str(round(auc,2))))
ax.plot([0,1],[0,1],linestyle='--',color='grey')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
plt.legend(fontsize=12)
plt.show()

end = time.time()
print('running time',str(end - start))
'''