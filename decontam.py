#TODO convert warning prints to warnings.warn call

#imports

from collections import Counter
from statistics import mean

import matplotlib.pyplot as plt
from numpy import add, log, multiply
from numpy import nan as NAN
from numpy import subtract
from scipy.stats import chi2, chi2_contingency, f, fisher_exact


#####
# helper functions

def stop(msg):
    print("ERROR: "+msg)
    exit(1)


def is_numeric(values):
    if isinstance(values,(int,float)):
        return True
    try:
        for val in values:
            float(val)
        return True
    except ValueError:
        return False


def fish_combine(vec):
    vec = [v for v in vec if v is not NAN]
    if any([v<0 or v>1 for v in vec]):
        stop("fish_combine expects values between 0 and 1.")
    p = multiply(vec)
    return chi2.cdf(-2*log(p), df=2*len(vec))


def fisher_pval(tab, alternative):
    excess = fisher_exact(tab, alternative="greater")[1] + fisher_exact(tab, alternative="less")[1] - 1
    pval = fisher_exact(tab, alternative=alternative)[1]
    pval -= excess / 2
    return pval


#####
# calculation functions

def isContaminantFrequency(freq,conc):
    numericNonZero = [f is not NAN and f > 0 for f in freq]
    log_f = [log(f) for f, valid in zip(freq,numericNonZero) if valid]
    log_c = [log(c) for c, valid in zip(conc,numericNonZero) if valid]

    if len(log_f) > 1:
        #test contam hypothesis (enforce fit line has slope -1)
        d1= add(log_f,log_c)
        beta1 = mean(d1)
        residuals1 = d1 - beta1
        SS1 = sum(residuals1 ** 2)

        #test null hypothesis (enforce fit line has slope 0)
        #d0 = log_f
        beta0 = mean(log_f)
        residuals0 = log_f - beta0
        SS0 = sum(residuals0 ** 2)

        #degrees of freedom
        dof = len(log_f) - 1

        #avoid divide by zero
        if SS0 > 0:
            #f test
            pval = f.cdf(SS1/SS0, dof, dof)
        else:
            pval = 1.0 #if SSO is a perfect match, fail to reject
    else:
        pval = NAN

    return pval


#TODO: fisher exact is working and matches the output from the R package.
# the chi2 method does NOT match, and since it is the default in auto mode
# this is not good.
def isContaminantPrevalence(freq, neg, method):
    isDetected = [f > 0 for f in freq]
    sumDetected = sum(map(int,isDetected))
    numSamples = sum([1 for n in neg if n is not NAN])
    numNegCtrl = sum([int(n) for n in neg if n is not NAN])
    
    if sumDetected > 1 and 0 < numNegCtrl < numSamples:
        tab = Counter(zip(neg,isDetected))
        #                 detected  not detected
        # negative sample    TT           TF
        # positive sample    FT           FF
        tab = [[tab[(True,True)],tab[(True,False)]],
               [tab[(False,True)],tab[(False,False)]]]
        
        # First entry (True,True) is the neg prevalence, so alternative is "greater"
        # dof = (rows-1)(cols-1) = 1*1 = 1
        if tab[0][1] + tab[1][1] == 0: #ASV Present in all samples
            #perfectly uncertain
            pval = 0.5

        elif method == "fisher":
            pval = fisher_pval(tab,alternative="greater")

        elif method == "chisq":
            pval = chi2_contingency(tab)[1]
        else: #auto
            try:
                # same as
                # x2stat, _ , dof, _ = chi2_contingency(tab)
                # pval  = chi2.sf(x2stat,dof)  #sf = 1-cdf
                pval = chi2_contingency(tab)[1]
            except:
                pval = fisher_pval(tab,alternative="greater")
    else:
        pval = NAN

    return pval


#####
# main functions

def isContaminant(seqtab, conc= None, neg = None, method = "auto", batch = None, batch_combine = "minimum", threshold = 0.1, normalize = True, detailed = True, prev_method = "auto"):
    ###############################
    # validate categorical inputs #
    ###############################

    assert method in ["auto", "frequency", "prevalence", "combined", "minimum", "either", "both"]
    assert batch_combine in ["minimum", "product", "fisher"]
    assert prev_method in ["auto", "fisher", "chisq"]
    

    ####################################
    # validate and format input matrix #
    ####################################

    if not all([is_numeric(row) for row in seqtab]):
        stop("seqtab must be a numeric matrix.")

    # convert all data entries to floats
    seqtab = [list(map(float,row)) for row in seqtab]

    #catch and remove zero-count samples
    hasZeroCount = [sum(row) == 0 for row in seqtab]
    if any(hasZeroCount):
        seqtab = [row for row, isZero in zip(seqtab,hasZeroCount) if not isZero]
        if conc is not None: conc = [row for row, isZero in zip(conc,hasZeroCount) if not isZero]
        if neg is not None: neg = [row for row, isZero in zip(neg,hasZeroCount) if not isZero]
        if batch is not None: batch = [row for row, isZero in zip(batch,hasZeroCount) if not isZero]
        print("WARNING: Removed ", sum(map(int,hasZeroCount)), " samples with zero total counts (or frequency).")

    numSamples = len(seqtab)
    numFeatures = len(seqtab[0])

    #normalize each row to sum == 1
    if normalize:
        sums = [sum(row) for row in seqtab]
        seqtab = [[val/sums[row] for val in seqtab[row]] for row in range(numSamples)]
    

    ######################################
    # automatically determine the method #
    ######################################

    if method == "auto":
        if conc is not None and neg is None: method = "frequency"
        elif conc is None and neg is not None: method = "prevalence"
        else: method = "combined"

    #establish which (freq/prev) test(s) will be run
    do_freq = False
    do_prev = False

    if method in ["frequency", "combined", "minimum", "either", "both"]: do_freq = True
    if method in ["prevalence", "combined", "minimum", "either", "both"]: do_prev = True
    

    ########################################
    # check for errors in the input format #
    ########################################

    #check conc and neg
    if do_prev and neg is None:
        stop("neg must be provided to perform prevalence-based contaminant identification.")
    if do_freq:
        if conc is None:
            stop("conc must be provided to perform frequency-based contaminant identification.")
        if not (is_numeric(conc) and all([c > 0 for c in conc])):
            stop("conc must be positive numeric.")
        if len(conc) != numSamples:
            stop("The length of conc must match the number of samples (the rows of seqtab).")
        if neg is None: neg = [False for _ in range(numSamples)]

    #check threshold
    if isinstance(threshold,(int,float)): threshold = [threshold]
    if is_numeric(threshold) and all([0 <= t <= 1 for t in threshold]):
        if method in ["either", "both"]:
            if len(threshold) == 1:
                print("WARNING: Using same threshold value for the frequency and prevalence contaminant identification.")
                threshold = threshold.extend(threshold)
        elif len(threshold) != 1:
            stop("threshold should be a single value.")
    else:
        stop("threshold must be a numeric value from 0 to 1 (inclusive).")

    #check batch
    if batch is None:
        batch = [1 for _ in range(numSamples)]
    if len(batch) != numSamples:
        stop("The length of batch must match the number of samples (the rows of seqtab).")
    tab_batch = Counter(batch)
    if min(tab_batch.values()) <= 1:
        stop("Some batches contain zero or one samples.")
    if min(tab_batch.values()) <= 4:
        print("WARNING: Some batches have very few (<=4) samples.")
    batch_keys = tab_batch.keys()


    #####################
    # loop over batches #
    #####################

    p_freqs = {i:[NAN for _ in range(numFeatures)] for i in batch_keys}
    p_prevs = {i:[NAN for _ in range(numFeatures)] for i in batch_keys}

    for bat in batch_keys:
        if do_freq:
            seqtab_subset = [seqtab[row] for row in range(numSamples) if batch[row] == bat and not neg[row]]
            conc_subset = [conc[row] for row in range(numSamples) if batch[row] == bat and not neg[row]]
            p_freqs[bat] = [isContaminantFrequency(col,conc_subset) for col in zip(*seqtab_subset)]

        if do_prev:
            seqtab_subset = [seqtab[row] for row in range(numSamples) if batch[row] == bat]
            neg_subset = [neg[row] for row in range(numSamples) if batch[row] == bat]
            p_prevs[bat] = [isContaminantPrevalence(col,neg_subset,method=prev_method) for col in zip(*seqtab_subset)]
    

    ####################
    # combine p-values #
    ####################

    p_freq = None #TODO should we use NaN here? if so, perhaps also in defaults for conc, etc.
    p_prev = None

    if batch_combine == "minimum":
        if do_freq:
            p_freq = [min([pval for pval in featurePValues if pval is not NAN]) if any([pval is not NAN for pval in featurePValues]) else NAN for featurePValues in zip(*p_freqs.values())]
        if do_prev:
            p_prev = [min([pval for pval in featurePValues if pval is not NAN]) if any([pval is not NAN for pval in featurePValues]) else NAN for featurePValues in zip(*p_prevs.values())]
    elif batch_combine == "product":
        if do_freq:
            p_freq = [multiply([pval for pval in featurePValues if pval is not NAN]) if any([pval is not NAN for pval in featurePValues]) else NAN for featurePValues in zip(*p_freqs.values())]
        if do_prev:
            p_prev = [multiply([pval for pval in featurePValues if pval is not NAN]) if any([pval is not NAN for pval in featurePValues]) else NAN for featurePValues in zip(*p_prevs.values())]
    elif batch_combine == "fisher":
        if do_freq:
            p_freq = [fish_combine([pval for pval in featurePValues if pval is not NAN]) if any([pval is not NAN for pval in featurePValues]) else NAN for featurePValues in zip(*p_freqs.values())]
        if do_prev:
            p_prev = [fish_combine([pval for pval in featurePValues if pval is not NAN]) if any([pval is not NAN for pval in featurePValues]) else NAN for featurePValues in zip(*p_prevs.values())]
    else:
        stop("Invalid batch_combine value.")


    #############################
    # calculate overall p-value #
    #############################

    pval = None
    if method == "frequency": pval = p_freq
    elif method == "prevalence": pval = p_prev
    elif method == "minimum": pval = [min(pvals) for pvals in zip(p_freq, p_prev)]
    elif method == "combined": pval = subtract(1, chi2.cdf(multiply(-2, multiply(log(p_freq), log(p_prev))),df=4))
    elif method in ["either","both"]: pval = [NAN for _ in range(numFeatures)]
    else: stop("Invalid method specified.")


    #########################################
    # determine which ASVs are contaminants #
    #########################################

    isC = None
    if method == "either":
        isC = [p_freq[p] < threshold[0] or p_prev[p] < threshold[1] for p in range(numFeatures)]
    elif method == "both":
        isC = [p_freq[p] < threshold[0] and p_prev[p] < threshold[1] for p in range(numFeatures)]
    else:
        isC = [pv < threshold[0] for pv in pval]

    #NAN pvals are not called contaminants
    isC = [b if b is not NAN else False for b in isC]


    #####################
    # make return value #
    #####################

    if not detailed: return isC

    rval = {}
    rval["freq"] = [mean(col) for col in zip(*seqtab)]
    rval["prev"] = [sum([int(c > 0) for c in col]) for col in zip(*seqtab)]
    rval["p_freq"] = p_freq
    rval["p_prev"] = p_prev
    rval["p"] = pval
    rval["contaminant"] = isC

    return rval


def isNotContaminant(seqtab, neg=None, threshold=0.5, normalize = True, detailed = False):
    contamData = isContaminant(seqtab,conc=None,neg=neg,method="prevalence",threshold=threshold,normalize=normalize,detailed=True)
    contamData["p_freq"] = subtract(1,contamData["p_freq"])
    contamData["p_prev"] = subtract(1,contamData["p_prev"])

    #calculate overall p-value
    pval = contamData["p_prev"]

    #make contaminant calls
    isNotC = [p < threshold if p is not NAN else False for p in pval]
    contamData["p"] = pval
    contamData["contaminant"] = None
    contamData["not_contaminant"] = isNotC

    #make return value
    if detailed:
        return contamData
    else:
        return isNotC