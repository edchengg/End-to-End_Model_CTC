import editdistance

def compute_cer(results):
    """
    Arguments:
        results (list): list of ground truth and
            predicted sequence pairs.

    Returns the CER for the full set.
    """
    dist = sum(editdistance.eval(label, pred)
                for label, pred in results)
    total = sum(len(label) for label, _ in results)
    return dist / total
