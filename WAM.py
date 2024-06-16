import numpy as np
import pandas as pd
import re
import xarray as xr
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from plotnine import*
import seaborn as sns
from sklearn.metrics import roc_curve,auc,roc_auc_score,confusion_matrix

donor_base_count = pd.DataFrame(np.zeros((9,4)),
                               columns = ['a','t','c','g'],
                               index=[i-3 for i in range(9)])
donor_pair_count = xr.DataArray(np.zeros((9,4,4)),
                               dims = ['position','forward','self'],
                               coords = [[i -3 for i in range(9)],
                               ['a','c','t','g'],
                               ['a','c','t','g']])
bg_pair_count = pd.DataFrame(np.zeros((4,4)),
                               columns = ['a','t','c','g'],
                               index=['a','t','c','g'])

donor_seqs = []
bg_seq = ''

folder_path = 'D:/data_mining/data/small'
folder_names = os.listdir(folder_path)

for now_file in tqdm(folder_names,desc='Reading files'):
    file_path = os.path.join(folder_path,now_file)
    with open(file_path,'r') as f:
        lines = f.readlines()
    donor_sites = re.findall('(\d+)(?=,)',lines[1])
    whole_seq = ''.join(lines[2:]).replace('\n','')

    for pos in donor_sites:
        donor_seqs.append(whole_seq[int(pos)-4:int(pos)+5])

        for i in range(9):
            base = whole_seq[int(pos)-4+i]
            donor_base_count.loc[i-3,base] += 1

        for i in range(8):
            base = whole_seq[int(pos)-3+i]
            forward_base = whole_seq[int(pos)-4+i]
            donor_pair_count.loc[i-2,forward_base,base] += 1

    bg_seq+=whole_seq

    for i in range(len(whole_seq)-1):
        base = whole_seq[i+1]
        forward_base = whole_seq[i]
        try:
            bg_pair_count.loc[forward_base,base]+=1
        except:
            continue

donor_matrix = donor_base_count/len(donor_seqs)


donor_pair_matrix =  donor_pair_count.copy()
for base in ['a','t','c','g']:
    donor_pair_matrix.loc[:,base] = donor_pair_matrix.loc[:,base]/donor_pair_matrix.loc[:,base].sum(axis=1)

donor_pair_matrix = donor_pair_matrix.fillna(0)
bg_seq.replace('n','')
p_neg = [bg_seq.count(base)/len(bg_seq) for base in ['a','t','c','g']]
neg_matrix = pd.DataFrame(np.tile(p_neg,(9,1)),
                          columns = ['a','t','c','g'],
                          index = [i-3 for i in range(9)])
neg_pair_matrix = bg_pair_count.copy()
for base in ['a','t','c','g']:
    neg_pair_matrix.loc[:,base] = neg_pair_matrix.loc[:,base]/neg_pair_matrix.loc[:,base].sum()



def WAM(seq):
    score = np.log((donor_matrix.loc[-3,seq[0]]+1)/neg_matrix.loc[-3,seq[0]])
    for i in range(8):
        score += np.log((donor_pair_matrix.loc[i-2,seq[i],seq[i+1]]+1)/neg_pair_matrix.loc[seq[i],seq[i+1]])
    
    return float(score)
###

test_site_seqs = []
test_normal_seqs = []

print('Loading testing data...')
test_file_path = 'D:/data_mining/data/small_test'
test_files = os.listdir(test_file_path)
for test_file in tqdm(test_files):
    with open(os.path.join(test_file_path,test_file), 'r') as f:
        text = f.readlines()
        test_site_positions = re.findall('(\d+)(?=,)', text[1])    # 提取位置
        seq = ''.join(text[2:]).replace('\n', '').lower()
        
        for position in test_site_positions:
            test_site_seqs.append(seq[int(position) - 4:int(position) + 5])
            
        for position in range(len(seq) - 8):
            test_normal_seq = seq[position:position+9]
            if test_normal_seq not in test_site_seqs and \
                set(test_normal_seq) == {'a', 't', 'c', 'g'}:
                test_normal_seqs.append(test_normal_seq)

pos_scores = [WAM(seq) for seq in test_site_seqs]
neg_scores = [WAM(seq) for seq in test_normal_seqs]
test_score = pos_scores+neg_scores
true_label = [1] * len(test_site_seqs) + [0] * len(test_normal_seqs)
pred_label = [1 if pred > 15.5 else 0 for pred in test_score]
cm = confusion_matrix(true_label, pred_label)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

auc = roc_auc_score(true_label,test_score)
fpr,tpr,thres = roc_curve(true_label,test_score,)
print('AUC ', auc)
fig,ax = plt.subplots(figsize=(10,8))
ax.plot(fpr,tpr,linewidth=2,label='SVM (AUC={})'.format(str(round(auc,2))))
ax.plot([0,1],[0,1],linestyle='--',color='grey')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
plt.legend(fontsize=12)
plt.show()



# scores = np.array(scores)
# plt.hist(scores, bins=17, color='skyblue', edgecolor='black') 
# plt.xlabel('Score')
# plt.ylabel('Number')
# plt.show()

#scores = np.array(scores)
# threshold = np.linspace(scores.min(),scores.max(),70)
# recall = [np.sum(scores > thr)/scores.shape for thr in threshold]
# recall = np.array(recall).reshape(-1)

# p = (ggplot(aes(x = threshold, y = recall)) +
#             geom_line(color = '#66ccff', size = 2) +
#             theme_bw()+
#             xlab('Recall') +
#             ylab('threshold') +
#             theme(axis_text_x = element_text(color = 'black')))

# print(p)

