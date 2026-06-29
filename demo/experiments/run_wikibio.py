import pickle
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from datasets import load_dataset
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
selfcheck = SelfCheckNLI(device=device)
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
dataset = dataset['evaluation']
label_mapping = {
    'accurate': 0.0,
    'minor_inaccurate': 0.5,
    'major_inaccurate': 1.0,
}
human_label_detect_False   = {}
human_label_detect_False_h = {}
human_label_detect_True    = {}
for i_ in range(len(dataset)):
    dataset_i = dataset[i_]
    idx = dataset_i['wiki_bio_test_idx']
    raw_label = np.array([label_mapping[x] for x in dataset_i['annotation']])
    human_label_detect_False[idx] = (raw_label > 0.499).astype(np.int32).tolist()
    human_label_detect_True[idx]  = (raw_label < 0.499).astype(np.int32).tolist()
    average_score = np.mean(raw_label)
    if average_score < 0.99:
        human_label_detect_False_h[idx] = (raw_label > 0.99).astype(np.int32).tolist()
len(human_label_detect_False), len(human_label_detect_True), len(human_label_detect_False_h)
def unroll_pred(scores, indices):
    unrolled = []
    for idx in indices:
        unrolled.extend(scores[idx])
    return unrolled
def get_PR_with_human_labels(preds, human_labels, pos_label=1, oneminus_pred=False):
    indices = [k for k in human_labels.keys()]
    unroll_preds = unroll_pred(preds, indices)
    if oneminus_pred:
        unroll_preds = [1.0-x for x in unroll_preds]
    unroll_labels = unroll_pred(human_labels, indices)
    assert len(unroll_preds) == len(unroll_labels)
    print("len:", len(unroll_preds))
    P, R, thre = precision_recall_curve(unroll_labels, unroll_preds, pos_label=pos_label)
    return P, R
def print_AUC(P, R):
    print("AUC: {:.2f}".format(auc(R, P)*100))
indices = [x['wiki_bio_test_idx'] for x in dataset] 
selfcheck_scores = {} # sentence-level scores
for i in tqdm(range(len(dataset))):
    x = dataset[i]
    idx = dataset[i]['wiki_bio_test_idx']
    selfcheck_scores[idx] = selfcheck.predict(
        sentences = x['gpt3_sentences'],           # list of sentences
        sampled_passages = x['gpt3_text_samples'], # list of sampled passages
    ).tolist()
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
# with human label, Detecting False
Prec, Rec = get_PR_with_human_labels(selfcheck_scores, human_label_detect_False, pos_label=1)
print("-----------------------")
print("SelfCheckGPT-NLI on WikiBio")
print_AUC(Prec, Rec)
arr = []
for v in human_label_detect_False.values():
    arr.extend(v)
random_baseline = np.mean(arr)
random_baseline
# with human label, Detecting Non-factual
plt.figure(figsize=(5.5, 4.5))
plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random Guessing') 
plt.plot(Rec, Prec, '-', label='SelfCheckGPT-NLI')
plt.legend()
plt.ylim(0.7,1.02)
plt.ylabel("Precision")
plt.xlabel("Recall")
# with human label, Detecting False
Prec, Rec = get_PR_with_human_labels(selfcheck_scores, human_label_detect_False_h, pos_label=1)
print("-----------------------")
print("SelfCheckGPT-NLI on WikiBio")
print_AUC(Prec, Rec)
arr = []
for v in human_label_detect_False_h.values():
    arr.extend(v)
random_baseline = np.mean(arr)
random_baseline
# with human label, Detecting Non-factual*
plt.figure(figsize=(5.5, 4.5))
plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random Guessing') 
plt.plot(Rec, Prec, '-', label='SelfCheckGPT-NLI')
plt.legend()
plt.ylabel("Precision")
plt.xlabel("Recall")
# with human label, Detecting True
Prec, Rec = get_PR_with_human_labels(selfcheck_scores, human_label_detect_True, pos_label=1, oneminus_pred=True)
print("-----------------------")
print("SelfCheckGPT-NLI on WikiBio")
print_AUC(Prec, Rec)
arr = []
for v in human_label_detect_True.values():
    arr.extend(v)
random_baseline = np.mean(arr)
random_baseline
# with human label, Detecting Non-factual*
plt.figure(figsize=(5.5, 4.5))
plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random Guessing') 
plt.plot(Rec, Prec, '-', label='SelfCheckGPT-NLI')
plt.legend()
plt.ylabel("Precision")
plt.xlabel("Recall")
