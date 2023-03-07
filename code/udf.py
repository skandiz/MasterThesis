# Power Law fit
def powerLawFit(funct, fit, powerlawExponents):
    for i in range(nDrops):
        powerlawFit = tp.utils.fit_powerlaw(funct[i], plot = False) 
        powerlawExponents[i] = powerlawFit.n.values 
        fit[i] = powerlawFit.A.values * np.array(funct.index)**powerlawExponents[i] 
    return fit, powerlawExponents

# Histogram fit
def fit_hist(y, bins_, distribution):
    if distribution == normal_distr:
        p0_ = [1., 0.]
    else:
        p0_ = [1.]

    bins_c = bins_[:-1] + np.diff(bins_) / 2

    bin_heights, _ = np.histogram(y, bins = bins_, density = True)
    ret, pcov = curve_fit(distribution, bins_c , bin_heights, p0 = p0_)
    ret_std = np.sqrt(np.diag(pcov))
    
    return ret, ret_std