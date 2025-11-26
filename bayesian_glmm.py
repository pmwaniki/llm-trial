from pathlib import Path
import pyro
import pandas as pd
import numpy as np
import torch
import statsmodels.formula.api as smf
import statsmodels.api as sm
from dotenv import dotenv_values
from pyro.infer import NUTS, MCMC
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from bayesian_models import logistic_glmm, logistic_glmm2,logistic_glmm_meta
import patsy
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import seaborn.objects as so

config=dotenv_values('.env')
result_folder=Path(config["RESULTS_FOLDER"])

data=pd.read_parquet(result_folder/'outcome_data_cleaned.parquet')
data=data.loc[~data['failure_num'].isna()].copy()
clin_arm={int(c):data.loc[data['clinician_num']==c,'arm'].iat[0] for c in data['clinician_num'].unique()}
hosp_map={int(h):data.loc[data['hospital_num']==h,'hosp_id'].iat[0] for h in data['hospital_num'].unique()}



#sanity check
model_freq=smf.glm("failure_num~arm_num+hosp_id",data=data,family=sm.families.Binomial()).fit()
model_freq.summary()

random = {"a": '0 + C(clinician_num)',"b": "0 + C(hospital_num)"}
model = BinomialBayesMixedGLM.from_formula(
               'failure_num~arm_num+age_category', random, data)
result = model.fit_vb()
result.summary()


#
mcmc_samples=1000
mcmc_burnin=100

# Pyro GLMM
y, x = patsy.dmatrices("failure_num~ 1 + arm_num", data, return_type="dataframe")
y,x=torch.tensor(y.values,dtype=torch.float32),torch.tensor(x.values[:,1:],dtype=torch.float32)
clinician=torch.tensor(data['clinician_num'],dtype=torch.int64)
center=torch.tensor(data['hospital_num'],dtype=torch.int64)
arm=torch.tensor([0. if clin_arm[i]=="Control" else 1. for i in range(clinician.max()+1)],dtype=torch.int64)


kernel = NUTS(logistic_glmm, full_mass=False)
mcmc = MCMC(kernel, num_samples=mcmc_samples, warmup_steps=mcmc_burnin, num_chains=1)
mcmc.run(x,clinician,center,y)

mcmc.summary(prob=0.95)

samples=mcmc.get_samples()



def plot_ranef(effects,xlab,ylab="Random Effect",hue_map=None,figsize=(12,12),ci=0.95,label_map=None):
    effects2 = effects.detach().cpu().numpy()
    intervals_=[]
    for j in range(effects.shape[1]):
        intervals_.append({'median':np.median(effects2[:,j]),'mean':np.mean(effects2[:,j]),
                           'lower':np.quantile(effects2[:,j],q=(1-ci)/2),'upper':np.quantile(effects2[:,j],q=ci+(1-ci)/2),
                           'lab': j if label_map is None else label_map[j],
                           'pos':j})
    intervals2=pd.DataFrame(intervals_)
    intervals2=intervals2.sort_values(by="median")
    intervals2['order']=range(intervals2.shape[0])
    if hue_map is not None:
        intervals2['hue']=intervals2['pos'].map(hue_map)
    format_dict={int(i):str(intervals2.loc[intervals2['order']==int(i),'lab'].iat[0]) for i in intervals2['order'].unique()}
    def formater_fun(a,b):
        return format_dict.get(a)
    # fig,ax=plt.subplots()
    # ax.scatter(intervals2['order'],intervals2['median'],label='mean')
    p=(
        so.Plot(intervals2, x="order", y="median",ymin="lower", ymax="upper", color= None if hue_map is None else "hue")
        .add(so.Dot())
        .add(so.Range())
        .scale(x=so.Continuous().tick(every=1).label(FuncFormatter(formater_fun)))
    )
    f=mpl.figure.Figure(figsize=figsize, dpi=300)
    # fig,ax=plt.subplots()

    # Render onto ax
    res=p.on(f).plot()
    ax = f.axes[0]
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc='upper right')
    # ax.set_xticks(list(format_dict.keys()))
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Rotate labels

    return f


f=plot_ranef(samples['b_cen'],xlab="Center",label_map=hosp_map)
f.savefig('/tmp/ranef_center.png')

f2=plot_ranef(samples['b_clin'],xlab="Clinician",hue_map=clin_arm,figsize=(20,10))
f2.savefig('/tmp/ranef_clinic.png')


kernel2 = NUTS(logistic_glmm2, full_mass=False)
mcmc2 = MCMC(kernel2, num_samples=mcmc_samples, warmup_steps=mcmc_burnin, num_chains=1)
mcmc2.run(x,clinician,center,arm,y)

mcmc2.summary(prob=0.95)

samples2=mcmc2.get_samples()

f_2=plot_ranef(samples2['b_cen'],xlab="Center",label_map=hosp_map)
f_2.savefig('/tmp/ranef_center2.png')

f2_2=plot_ranef(samples2['b_clin'],xlab="Clinician",hue_map=clin_arm,figsize=(20,10))
f2_2.savefig('/tmp/ranef_clinic2.png')



kernel_meta = NUTS(logistic_glmm_meta, full_mass=False)
mcmc_meta = MCMC(kernel_meta, num_samples=mcmc_samples, warmup_steps=mcmc_burnin, num_chains=1)
mcmc_meta.run(x,clinician,center,y)

mcmc_meta.summary(prob=0.95)
samples_meta=mcmc_meta.get_samples()
f=plot_ranef(samples_meta['b_cen_decentered'],xlab="Center",label_map=hosp_map)
f.savefig('/tmp/ranef_center.png')
