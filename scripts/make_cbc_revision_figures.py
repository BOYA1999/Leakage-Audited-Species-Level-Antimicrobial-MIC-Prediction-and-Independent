import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import make_jcim_model_benchmark_figure as bench
import make_species_mic_figures as legacy

parser = argparse.ArgumentParser(description='Render CBC display allocations from frozen tables.')
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
TABLES, RESULTS, OUT = args.data / 'tables', args.data / 'evidence', args.out
OUT.mkdir(parents=True, exist_ok=True)
plt.style.use(Path(__file__).with_name('cbc_academic.mplstyle'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.grid': False})
colors = ['#87959C', '#476F88', '#9D6557']
bench.COLORS.update({'Morgan': colors[1], 'Multi-view': colors[2], 'Chemprop': '#608677', 'MolE': '#A19467', 'MoLFormer': '#82728E'})
def save(fig, name):
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{extension}', dpi=450, bbox_inches='tight', facecolor='white')
    plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 4.15))
ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
def box(x, y, w, h, title, detail, fill='#F0F3F4'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.009', facecolor=fill, edgecolor='#9BA8AE', linewidth=0.8))
    ax.text(x + w/2, y + h*0.76, title, ha='center', va='center', fontsize=8.7, weight='bold')
    ax.text(x + w/2, y + h*0.33, detail, ha='center', va='center', fontsize=8.0, linespacing=1.25)
def arrow(x1,y1,x2,y2):
    ax.annotate('', (x2,y2), (x1,y1), arrowprops={'arrowstyle':'->','color':'#788B94','lw':1.1})
box(.015,.73,.27,.23,'ChEMBL 34 MIC','720,492 source records')
box(.365,.73,.27,.23,'Exact measurements','Relation, unit, validity filters\n442,271 records')
box(.715,.73,.27,.23,'Species-level endpoints','Taxonomy, structure curation\n214,068 pairs · 48 species')
arrow(.29,.845,.355,.845); arrow(.64,.845,.705,.845)
box(.21,.38,.58,.23,'Benchmark and generalization audit','Five scaffold partitions · fixed model comparisons\nSpecies median · trees · embeddings · D-MPNN','#EDF2EE')
arrow(.85,.72,.72,.62)
for x,title,detail in [(.015,'Document-year test','Train ≤2018\nTest 2021–2023\nTraining-era median control'),(.365,'Held-out species','48 folds · compound reuse\nKnown broad-class control'),(.715,'Joint external ranking','Two matched species\nMolecular weight control')]:
    box(x,.015,.27,.24,title,detail)
    arrow(.5,.375,x+.135,.265)
save(fig,'fig1_study_design')

summary = pd.read_csv(TABLES / 'jcim_model_benchmark_summary.csv')
paired = pd.read_csv(TABLES / 'jcim_model_paired_differences.csv')
cost = pd.read_csv(TABLES / 'jcim_complexity_summary.csv')
bench.check_inputs(summary, paired, cost)
fig, axes = plt.subplots(2,1,figsize=(8,7.2))
fig.subplots_adjust(left=.27,right=.98,bottom=.11,top=.94,hspace=.70)
bench.performance_panel(axes[0],summary); bench.paired_panel(axes[1],paired)
save(fig,'fig2_model_comparison')
fig, axes = plt.subplots(1,2,figsize=(8,3.5))
bench.cost_panel(axes[0],cost,summary,'fit_seconds_mean','AFit time','Recorded fit time (s; log scale)')
bench.cost_panel(axes[1],cost,summary,'serialized_object_bytes_mean','BSerialized size','Recorded bytes (log scale)')
fig.tight_layout(w_pad=2.5)
save(fig,'figS1_recorded_cost')

temporal = pd.read_csv(RESULTS / 'temporal_with_median.csv')
fig, axes = plt.subplots(2,2,figsize=(8,5.3))
axes[0,0].axis('off')
axes[0,0].set_title('A  Document-year cohorts',weight='bold')
for y,label,n in [(.76,'Train through 2018','179,208 pairs · 51,001 compounds'),(.46,'Validation 2019–2020','16,461 pairs · 6,275 compounds'),(.16,'Test 2021–2023','13,999 pairs · 5,359 compounds')]:
    axes[0,0].text(.03,y,label,weight='bold',fontsize=10.3)
    axes[0,0].text(.03,y-.12,n,fontsize=9.7,color='#536273')
models=['species_median','morgan','multiview']; labels=['Species median','Morgan','Multi-view']
for i,model in enumerate(models):
    sub=temporal[temporal.model.eq(model)].set_index('scope')
    axes[0,1].bar(np.arange(2)+(i-1)*.24,[sub.loc['pooled','mae'],sub.loc['macro','mae']],.24,color=colors[i],label=labels[i])
    axes[1,0].bar(np.arange(2)+(i-1)*.24,[sub.loc['macro','within_one_dilution'],sub.loc['macro','within_two_dilutions']],.24,color=colors[i],label=labels[i])
axes[0,1].set(xticks=[0,1],xticklabels=['Pooled pairs','Equal species'],ylabel='MAE of log₂ MIC',ylim=(0,4.3),title='B  Error on the same test pairs')
axes[0,1].legend(fontsize=8.5,ncol=1,loc='upper left')
axes[1,0].set(xticks=[0,1],xticklabels=['Within 1 dilution','Within 2 dilutions'],ylabel='Equal-species accuracy',ylim=(0,.6),title='C  Dilution accuracy')
support=pd.read_csv(TABLES/'jcim_temporal_support_sensitivity.csv')
sub=support[support.model.eq('multiview')].sort_values('minimum_test_pairs')
x=np.arange(len(sub))
axes[1,1].plot(x,sub.mae_macro,'o-',color=colors[2])
for i,row in enumerate(sub.itertuples()): axes[1,1].annotate(f'n={row.species}',(i,row.mae_macro),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
axes[1,1].set(xticks=x,xticklabels=sub.minimum_test_pairs,ylim=(2.22,2.83),xlabel='Minimum test pairs per species',ylabel='Multi-view macro MAE',title='D  Species-support sensitivity')
fig.tight_layout(h_pad=2.2,w_pad=2.0)
save(fig,'fig3_temporal_controls')

joint=pd.read_csv(TABLES/'joint_external_paired_differences.csv')
mw=pd.read_csv(RESULTS/'joint_minus_molecular_weight.csv')
fig,axes=plt.subplots(2,1,figsize=(8.1,6.8))
for ax,metric,letter in zip(axes,['auroc','ap'],['A','B']):
    ticklabels=[]; positions=[]
    for start,organism,short,color in [(0,'Escherichia coli','E. coli',colors[1]),(5,'Clostridioides difficile','C. difficile',colors[2])]:
        sub=joint[joint.organism.eq(organism)&joint.cohort.eq('primary')&joint.metric.eq(metric)].set_index('comparison')
        rows=[sub.loc[c] for c in ['joint:multiview_minus_joint:morgan','joint:multiview_minus_joint:morgan_maccs','joint:multiview_minus_single:multiview']]
        rows.append(mw[mw.organism.eq(organism)&mw.cohort.eq('primary')&mw.metric.eq(metric)&mw.model.eq('multiview')].iloc[0])
        for k,(name,row) in enumerate(zip(['joint Morgan','joint Morgan + MACCS','matched single multi-view','molecular weight only'],rows)):
            y=start+k; positions.append(y); ticklabels.append(short+' — '+name)
            ax.errorbar(row['difference'],y,xerr=[[row['difference']-row.ci_low],[row.ci_high-row['difference']]],fmt='o',color=color,ms=5,capsize=2.5)
    ax.set_yticks(positions,ticklabels,fontsize=10)
    ax.invert_yaxis(); ax.axvline(0,color='#889198',lw=.8,ls='--')
    ax.set_title(letter+'  '+metric.upper()+' differences',weight='bold')
    ax.set_xlabel('Joint multi-view minus comparator')
fig.tight_layout(h_pad=2)
save(fig,'fig6_joint_external')

legacy.MODEL_COLORS.update(dict(zip(['morgan','multiview','mole','molformer'],[colors[1],colors[2],'#A19467','#82728E'])))
original_headline=legacy.headline
legacy.headline=lambda fig,text: original_headline(fig,'Separately trained AntiMicrobial-KG broad classifiers')
primary_broad_n = int(pd.read_csv(TABLES/'maier_external_metrics.csv').query("cohort == 'primary_exact_key_nonoverlap'").iloc[0].n)
def broad_layout(fig):
    for ax,title in zip(fig.axes,[f'Primary cohort (n = {primary_broad_n:,})','Paired representation differences','Primary-cohort top-k precision','Cohort sizes after exclusions']):
        ax.set_title(title,loc='left',fontsize=9.5)
    fig.axes[0].set_ylim(0,1.1)
    fig.axes[0].legend(fontsize=8,ncol=2,loc='upper right')
    for label in list(fig.axes[2].texts):
        if label.get_position()[0] == 102: label.remove()
    fig.axes[2].legend(fig.axes[2].lines[:3],['Morgan','Multi-view','Prevalence'],fontsize=7,loc='center left',bbox_to_anchor=(0,.32))
    for label in fig.texts: label.set_fontsize(11)
    fig.subplots_adjust(left=.10,right=.98,bottom=.11,top=.82,wspace=.55,hspace=.7)
legacy.finish_grid=broad_layout
legacy.figure7(TABLES,OUT,OUT/'source_data')
for extension in ['png','pdf','svg']:
    (OUT/f'fig7_maier_broad_transfer.{extension}').replace(OUT/f'figS3_separate_broad.{extension}')
sim=pd.read_csv(TABLES/'maier_similarity_metrics.csv')
bins=['<0.30','0.30-0.50','0.50-0.70','>=0.70']; x=np.arange(4)
fig,axes=plt.subplots(1,2,figsize=(8,3.5))
comp=sim[sim.model.eq('morgan')].set_index('similarity_bin').loc[bins]
axes[0].bar(x,comp.n-comp.positives,color='#D5DDE0',label='Inactive')
axes[0].bar(x,comp.positives,bottom=comp.n-comp.positives,color=colors[2],label='Active')
axes[0].set(ylabel='Compounds',title='A  Primary-cohort composition'); axes[0].legend(fontsize=8)
for model in ['morgan','multiview','mole','molformer']:
    axes[1].plot(x,sim[sim.model.eq(model)].set_index('similarity_bin').loc[bins].auprc,'o-',label=legacy.MODEL_LABELS[model],color=legacy.MODEL_COLORS[model])
axes[1].plot(x,comp.prevalence,'s--',color='#536273',label='Prevalence')
axes[1].set(ylabel='Average precision / prevalence',ylim=(0,1),title='B  Separate broad-classifier ranking'); axes[1].legend(fontsize=8,ncol=2)
for ax in axes: ax.set_xticks(x,bins,fontsize=9); ax.set_xlabel('Global broad-training Morgan similarity')
fig.tight_layout(w_pad=2)
save(fig,'figS4_separate_similarity')
(OUT/'figure_sources.json').write_text(json.dumps({'main':['fig1_study_design','fig2_model_comparison','fig3_temporal_controls','existing fig4_loso_transfer','existing fig6_conformal_uncertainty','fig6_joint_external'],
    'source_tables':'paper/tables/species_mic_v2','new_controls':'paper/review/evidence_20260912/results','script':'paper/scripts/make_cbc_revision_figures.py'},indent=2),encoding='utf-8')
