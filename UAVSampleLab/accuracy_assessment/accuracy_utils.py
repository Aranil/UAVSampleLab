"""
Collection of function to calculate accuracy metrics
was used for accuracy matrices - exported from eCognition
"""
import numpy as np
import pandas as pd



tick_labels = {
    'bare_soil': 'bare soil',
    'dry_crop': 'dry crop',
    'dry_lodged_crop': 'dry lodged crop',
    'flowering_crop': 'flowering crop',
    'ripening_crop': 'ripening crop',
    'vital_crop': 'vital crop',
    'vital_lodged_crop': 'vital lodged crop',
    'weed_infestation': 'weed infestation'
    # 'micro-average': 'Micro Average',
}




def confusion_metrics(dfA):
    """
    Computes detailed classification metrics from a confusion matrix DataFrame.

    Parameters:
    dfA: pandas.DataFrame

    Returns:
    df_report: DataFrame with per-class metrics
    """
    labels = dfA.index.tolist()
    conf_matrix = dfA.values
    n_classes = len(labels)

    TP = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TP
    FN = conf_matrix.sum(axis=1) - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP)!=0)
        recall = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN)!=0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(TP, dtype=float), where=(precision + recall)!=0)
        accuracy = np.divide(TP + TN, TP + TN + FP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + TN + FP + FN)!=0)

        false_discovery_rate = 1 - precision
        false_negative_rate = np.divide(FN, FN + TP, out=np.zeros_like(FN, dtype=float), where=(FN + TP)!=0)
        false_positive_rate = np.divide(FP, FP + TN, out=np.zeros_like(FP, dtype=float), where=(FP + TN)!=0)
        negative_predictive_value = np.divide(TN, TN + FN, out=np.zeros_like(TN, dtype=float), where=(TN + FN)!=0)
        positive_predictive_value = precision
        sensitivity = recall
        specificity = np.divide(TN, TN + FP, out=np.zeros_like(TN, dtype=float), where=(TN + FP)!=0)
        true_negative_rate = specificity
        true_positive_rate = recall
        omission_error = np.divide(FN, FN + TN, out=np.zeros_like(FN, dtype=float), where=(FN + TN)!=0)

    df_report = pd.DataFrame({
        'class': labels,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_discovery_rate': false_discovery_rate,
        'false_negative_rate': false_negative_rate,
        'false_positive_rate': false_positive_rate,
        'negative_predictive_value': negative_predictive_value,
        'positive_predictive_value': positive_predictive_value,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'true_negative_rate': true_negative_rate,
        'true_positive_rate': true_positive_rate,
        'user_accuracy': recall * 100,
        'producer_accuracy': precision * 100,
        'omission_error': omission_error * 100,
        'support': TP + FN  # how many times this class appeared in ground truth
    })

    return df_report.set_index('class')


def calculate_precision(TP, FP):
    """
    user

    Parameters
    ----------
    TP: float
        True positive value
    FP: float
        False positive value

    Returns
    -------
    precision
    """
    return TP / (TP + FP)


def calculate_recall(TP, FN):
    """
    producer

    Parameters
    ----------
    TP: float
        True positive value
    FN: float
        False negative value

    Returns
    -------
    recall
    """
    return TP/(TP+FN)


def F_measure(recall, precision):
    """
    f1 measure

    Parameters
    ----------
    recall: float
    precision:float

    Returns
    -------

    """
    return 2*(recall * precision) / (recall + precision)


def calculate_accuracy(sensitivity, specificity, TP, TN):
    """
    Parameters
    ----------
    sensitivity: float
        value of sensitivity in %
    specificity: float
        value of specificity in %
    TP: float
        True positive value
    TN: float
        True negative value

    Returns
    -------

    """
    return ( (sensitivity * TP) + (specificity + TN) ) / (TP + TN)


def classification_report(A, classes):
    '''
    calculates TP, TN, FP, FN & Total amount of used Samples from accuracy matrix

    A: np.array
        confusion matrix
    classes: list
        list of classes - labels for a matrix

        classes = [
                'bare_soil',
                'vital_crop',
                'vital_lodged_crop',
                'flowering_crop',
                'dry_crop',
                'dry_lodged_crop',
                'ripening_crop',
                'weed_infestation'
               ]

    source: https://stackoverflow.com/questions/48100173/how-to-get-precision-recall-and-f-measure-from-confusion-matrix-in-python
    '''
    # confusion matrix
    #A = df_subset.iloc[:, 1:-2].astype(int).to_numpy()
    #print(A)

    TP = np.diag(A) #
    FP = np.sum(A, axis=0) - TP
    FN = np.sum(A, axis=1) - TP

    num_classes = len(classes)
    TN = []
    for i in range(num_classes):
        temp = np.delete(A, i, 0)  # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))

    TotalSamp = []
    # check
    n = 10000
    for i in range(num_classes):
        #print(i)
        #print(TP[i] + FP[i] + FN[i] + TN[i] == n)

        #TotalSamples = TP + FP + FN + TN
        #Total_Positive_Samples = TP + FN
        #Total_Negative_Samples = FP + TN

        # Calculate the total number of samples per class
        TotalSamp.append(TP[i] + FP[i])
        print(TP[i], FP[i], FN[i], TN[i], classes[i])

       # TotalSamp[classes[i]] = TP[i] + FP[i]
        #print(TP[i], FP[i], FN[i], TN[i], TotalSamp[classes[i]], TP[i] + FP[i] + FN[i] + TN[i])

    return TP, FP, FN, TN,  np.asarray(TotalSamp)



# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)


# as an alternative to pkg dissaray, if want to calculate accuracy from matrix
'''
# Example of the usage !

#A = dff.reset_index().iloc[:, 1:].astype(int).to_numpy()
TP, FP, FN = classification_report(A)

precision = calculate_precision(TP, FP) # user
recall = calculate_recall(TP, FN) # producer
f_meas = F_measure(recall, precision)

print(precision)
print(recall)
print(f_meas)
'''
