import os
import json
from random import randint
from sklearn.metrics import roc_curve, auc

def monte_carlo_metrics(results, num_trials, tail_value, accuracy_only=True):
    '''
    Compute confidence intervals for accuracy, sensitivity, specificity, ppv, and npv on a set of data using monte carlo trials.
    '''
    tp, tn, fp, fn = get_conf_matrix(results)
    sample_size = tp + tn + fp + fn
    
    metrics_dict = {'accuracy':[],'sensitivity':[],'specificity':[],'ppv':[],'npv':[]}
    if accuracy_only:
        metrics_dict = {'accuracy':[]}
    CIs = dict.fromkeys(metrics_dict.keys())
    
    for x in range(num_trials):
        sample_tp, sample_tn, sample_fp, sample_fn = [0, 0, 0, 0]
        for y in range(sample_size):
            instance = randint(0, sample_size-1)
            if instance < tp:
                sample_tp += 1
            elif instance < tp + tn:
                sample_tn += 1
            elif instance < tp + tn + fp:
                sample_fp += 1
            else:
                sample_fn +=1
        metrics_dict['accuracy'].append((sample_tp + sample_tn) / sample_size)
        if not accuracy_only:
            metrics_dict['sensitivity'].append(sample_tp / (sample_tp + sample_fn))
            metrics_dict['specificity'].append(sample_tn / (sample_tn + sample_fp))
            metrics_dict['ppv'].append(sample_tp / (sample_tp + sample_fp))
            metrics_dict['npv'].append(sample_tn / (sample_tn + sample_fn))
    
    for metric in metrics_dict.keys():
        metrics_dict[metric].sort()
        CIs[metric] = [metrics_dict[metric][int(tail_value * num_trials)], metrics_dict[metric][int((1 - tail_value)* num_trials)]]
    return CIs

def monte_carlo_auc(y_true, y_pred, num_trials, tail_value):
    '''
    Compute confidence intervals for AUC on a set of data using monte carlo trials.
    '''
    sample_size = len(y_true)
    roc_samples = []
    
    for x in range(num_trials):
        sample_y_true, sample_y_pred = [], []
        for y in range(sample_size):
            i = randint(0, sample_size-1)
            sample_y_true.append(y_true[i])
            sample_y_pred.append(y_pred[i])
        fpr, tpr, thresh = roc_curve(sample_y_true, sample_y_pred, pos_label='Cancer')
        roc_samples.append(auc(fpr, tpr))
    roc_samples.sort()
    return [roc_samples[int(tail_value * num_trials)], roc_samples[int((1-tail_value) * num_trials)]]

def get_conf_matrix(results):
    return len(results['tp']), len(results['tn']), len(results['fp']), len(results['fn'])

def print_metrics(results):
    tp, tn, fp, fn = get_conf_matrix(results)
    print("True positive: ", tp)
    print("True negative: ", tn)
    print("False positive: ", fp)
    print("False negative: ", fn)
    print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
    if tp + fn != 0:
        print("Sensitivity: ", tp / (tp + fn))
    if tn + fp != 0:
        print("Specificity: ", tn / (tn + fp))
    if tp != 0 and fp != 0:
        print("PPV: ", tp / (tp + fp))
    if tn != 0 and fn != 0:
        print("NPV: ", tn / (tn + fn))

def aggregate(name, patches, gt_threshold, inf_threshold):
    '''
    Aggregates ROI level results to FOV level results, using a weighted median aggregation method. 
    '''
    gt_per = len([patch for patch in patches if patch['object_class'] == 'Cancer']) / len(patches)
    gt = 'Cancer' if gt_per > gt_threshold else 'Benign'

    n_max = int(len(patches) * inf_threshold)
    inf = sorted([patch['conf'][1] for patch in patches],reverse=True)[n_max]
    return {'id': name, 'object_class': gt, 'conf': [1 - inf, inf]}

def aggregate_max(name, patches):
    '''
    Aggregates ROI level results to FOV level results, taking the FOV level result as the max of the ROI level results.
    '''
    gt = 'Benign'
    for p in patches:
        if p['object_class'] == 'Cancer':
            gt = 'Cancer'
            break
    inf = max([p['conf'][1] for p in patches])
    return {'id': name, 'object_class': gt, 'conf': [1- inf, inf]}

def filter_by(list_of_dicts, key_name, val, exclude):
    '''
    Takes in a list of dicts. If exclude, returns a subset of the list which has key_name != val.
    If not exclude, returns a subset of the list which has key_name == val.
    '''
    if exclude:
        return [d for d in list_of_dicts if d[key_name] != val]
    else:
        return [d for d in list_of_dicts if d[key_name] == val]

def uniq(in_list):
    '''
    Gets the unique values in a list.
    '''
    l = []
    for i in in_list:
        if not i in l:
            l.append(i)
    return l

def inf_gt_preprocessing(inf_dir, gt_dir):
    # load and parse inference files from inf_dir
    inf = {}
    for slide in os.listdir(inf_dir):
        for file in os.listdir(os.path.join(inf_dir, slide)):
            with open(os.path.join(os.path.join(inf_dir, slide), file)) as f:
                for item in json.loads(f.read()):
                    inf.setdefault(slide, {})
                    inf[slide].setdefault(item['input_file'], []).append(item)
                    
    # load and parse ground truth files from gt_dir
    gt = {}
    for slide in os.listdir(gt_dir):
        for file in os.listdir(os.path.join(gt_dir, slide)):
            with open(os.path.join(os.path.join(gt_dir, slide), file)) as f:
                temp = json.loads(f.read())[0]
                gt.setdefault(slide, {})
                gt[slide][temp['input_file']] = temp['chips']
                
    # reorder inference to align with ground truth format ordering.
    for slide in inf.keys():
        for fov in inf[slide].keys():
            inf[slide][fov] = sorted(inf[slide][fov], key=lambda k: (k['points'][0]['x'], k['points'][0]['y']))

    # merge information from inf and gt. this changes gt in place. 
    for slide in gt.keys():
        for fov in gt[slide].keys():
            for i, _ in enumerate(gt[slide][fov]):
                gt[slide][fov][i]['conf'] = inf[slide][fov][i]['conf']
                gt[slide][fov][i]['points'] = inf[slide][fov][i]['points']
                gt[slide][fov][i].pop('frame_id')
                
    # remove undefined patches from fovs and undefined fovs from slide dict. this changes gt in place. also removes desmoplasia
    for slide in gt.keys():
        undefined = []
        for fov in gt[slide].keys():
            defined = []
            for patch in gt[slide][fov]:
                if patch['object_class'] != 'Undefined' and patch['short_name'] != 'Desmoplasia':
                    defined.append(patch)
            gt[slide][fov] = defined
            if not defined:
                undefined.append(fov)
        for f in undefined:
            gt[slide].pop(f)

    # replace patches with short_name 'met BrCA' with 'BrCA'.
    for slide in gt.keys():
        for fov in gt[slide].keys():
            for patch in gt[slide][fov]:
                if patch['short_name'] == 'met BrCA':
                    patch['short_name'] ='BrCA'
    return inf, gt