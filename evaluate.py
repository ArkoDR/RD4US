import os
import cv2
import json
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, confusion_matrix
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from skimage.util import img_as_ubyte
from dataset import USDataset
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2


def cal_metrics(results, labels, dynamic):
    """
    Calculate classification metrics and ROC curve.
    
    Args:
        results (array-like): Model prediction scores/confidence values.
        labels (array-like): Ground truth labels (1 for positive/anomaly, 0 for negative/normal).
        dynamic (bool or float): If True, uses all ROC thresholds; if float, uses that fixed threshold.
    
    Returns:
        dict: Contains metrics, ROC curve data, and AUC.
            - 'metrics': [best_accuracy, specificity, sensitivity, precision, NPV]
            - 'roc': [fpr, tpr]
            - 'roc_auc': Area under ROC curve
    """
    best_acc = 0
    acc = spe = sen = ppv = npv = 0
    results = np.array(results)
    labels = np.array(labels)
    # Invert labels for auc calculation
    labels = -1 * labels + 1
    fpr, tpr, thresholds = roc_curve(labels, results, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    # Restore original label
    labels = -1 * labels + 1

    if dynamic == True:
        splits = thresholds
    else:
        splits = [dynamic]
    for split in splits:
        temp = np.where(results > split, 0, 1)
        acc = accuracy_score(labels, temp)
        if acc > best_acc:
            best_acc = acc
            tn, fp, fn, tp = confusion_matrix(labels, temp).ravel()
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0
            sen = recall_score(labels, temp)
            ppv = precision_score(labels, temp)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {"metrics": [best_acc, spe, sen, ppv, npv], "roc": [fpr, tpr], "roc_auc": roc_auc}


def score_function(a, b, c, d):
    """
    Compute anomaly scores based on cosine similarity differences.
    
    For each set of feature pairs, calculate:
        score = max(cosine_sim(a,b) - cosine_sim(c,d), 0)
    
    Args:
        a, b, c, d (list of torch.Tensor): Output feature tensors.
    Returns:
        numpy.ndarray: Anomaly scores for each sample.
    """
    cos = torch.nn.CosineSimilarity()
    scores = []
    for item in range(len(a)):
        cos1 = cos(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1))
        cos2 = cos(c[item].view(c[item].shape[0], -1), d[item].view(d[item].shape[0], -1))
        temp = torch.clamp(cos1 - cos2, min=0)
        scores.append(torch.mean(temp).cpu().numpy())
    return np.array(scores)


def show_anomaly_map(amap, name = "amap", pic = None):
    """
    Display anomaly heatmap.
    
    Args:
        amap (numpy.ndarray): Anomaly map (2D array).
        name (str): Window name for the anomaly map.
        pic (numpy.ndarray, optional): Original image to display alongside.
    """
    amap = img_as_ubyte(amap / np.max(amap))
    amap = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
    cv2.imshow(name, amap)
    if pic is not None:
        cv2.imshow("pic", pic)
    cv2.waitKey(0)


def cal_anomaly_map(fs_list, ft_list, out_size=256, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map

    return anomaly_map, a_map_list


def evaluation(encoder, bn_layer, decoder, dataloader, device, preview = False, dynamic = False):
    """
    Main evaluation function.
    
    Args:
        encoder (nn.Module): Feature encoder network.
        bn_layer (nn.Module): Bottleneck layer.
        decoder (nn.Module): Reconstruction decoder network.
        dataloader (DataLoader): Test/validation data loader.
        device (torch.device): Computation device.
        preview (bool): Whether to visualize anomaly maps during evaluation.
        dynamic (bool or float): Threshold selection mode for metrics calculation.
    
    Returns:
        dict: Evaluation results.
    """
    encoder.eval()
    bn_layer.eval()
    decoder.eval()
    results = []
    labels = []
    with torch.no_grad():
        for pic_a, pic_e, pic_k, label in tqdm.tqdm(dataloader):
            pic_a = pic_a.to(device)
            pic_e = pic_e.to(device)
            pic_k = pic_k.to(device)
            inputs_a = encoder(pic_a)
            inputs_e = encoder(pic_e)
            inputs_k = encoder(pic_k)

            inputs_a_e = inputs_a[:1] + inputs_e[1:]
            inputs_e_a = inputs_e[:1] + inputs_a[1:]
            inputs_a_k = inputs_a[:1] + inputs_k[1:]
            inputs_e_k = inputs_e[:1] + inputs_k[1:]

            bn_a_e = bn_layer(inputs_a_e)
            bn_e_a = bn_layer(inputs_e_a)
            bn_a_k = bn_layer(inputs_a_k)
            bn_e_k = bn_layer(inputs_e_k)

            outputs_a_e = decoder(bn_a_e)
            outputs_e_a = decoder(bn_e_a)
            outputs_a_k = decoder(bn_a_k)
            outputs_e_k = decoder(bn_e_k)

            scores = score_function(inputs_a + inputs_e, outputs_a_e + outputs_e_a, inputs_a + inputs_e, outputs_a_k + outputs_e_k)
            ano_score = scores[2] + scores[5]
           
            results.append(ano_score)

            labels.append(label.cpu().numpy())

            if preview:
                anomaly_map_1, a_map_list = cal_anomaly_map(inputs_a + inputs_e, outputs_a_e + outputs_e_a, amap_mode='a')
                anomaly_map_2, a_map_list = cal_anomaly_map(inputs_a + inputs_e, outputs_a_k + outputs_e_k, amap_mode='a')
                anomaly_map = np.clip(anomaly_map_2 - anomaly_map_1, 0, None)
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)

    res = cal_metrics(results, labels, dynamic)

    return res


def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, metavar="", help="Path of dataset.")
    parser.add_argument("--model_path", type=str, required=True, metavar="", help="Path of model.")
    parser.add_argument("-s", "--size", type=int, default=256, metavar="", help="")
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    test_path = args.test_dir
    ckp_path = args.model_path
    test_data = USDataset(root=test_path, mode="test", size=args.size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn_layer = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn_layer = bn_layer.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    bn_layer.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])

    res = evaluation(encoder, bn_layer, decoder, test_dataloader, device, True, True)
    metrics = res["metrics"]
    roc = res["roc"]
    roc_auc = res["roc_auc"]
    acc, spe, sen, ppv, npv = metrics

    print('Accuracy{:.4f}, Specificity{:.4f}, Sensitivity(Recall){:.4f}, PPV(Precision){:.4f}, NPV{:.4f}, AUC{:.4f}'.format(acc, spe, sen, ppv, npv, roc_auc))

    return acc


if __name__ == "__main__":
    test()