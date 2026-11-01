"""Draw a vector audit matrix from source-bound natural-module classifications."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from check_natural_modules import derive,ROOT
from plot_figures import configure_fonts
def draw(out):
    out.mkdir(parents=True,exist_ok=True);data=derive(ROOT/'results/natural-modules');configure_fonts()
    modules=list(dict.fromkeys(r['module'] for r in data['attempt_categories']))
    categories={'admitted':0,'original_checks_failed':1,'edited_checks_failed':1,'original_validation_unavailable':2,'edited_validation_unavailable':2,'unsupported':3}
    symbols=['A','F','U','S'];colors=['#c6dcca','#dfb5aa','#d6d9de','#eee0b0']
    matrix=[[None]*4 for m in modules]
    for r in data['attempt_categories']:matrix[modules.index(r['module'])][r['key']]=categories[r['reason']]
    fig,ax=plt.subplots(figsize=(5.65,3.25));ax.imshow(matrix,cmap=ListedColormap(colors),vmin=0,vmax=3,aspect='auto')
    ax.set_xticks(range(4),['K1','K2','K3','K4']);ax.set_yticks(range(8),modules)
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=5)
    for y,row in enumerate(matrix):
        for x,c in enumerate(row):ax.text(x,y,symbols[c],ha='center',va='center',fontsize=9)
    for s in ax.spines.values():s.set_visible(False)
    for x in range(5):ax.axvline(x-.5,color='white',linewidth=1)
    for y in range(9):ax.axhline(y-.5,color='white',linewidth=1)
    labels=['A: admitted','F: task check failed','U: validation unavailable','S: embedding abstained']
    handles=[Patch(facecolor=c,edgecolor='#777777',linewidth=.3,label=l) for c,l in zip(colors,labels)]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(.53,-.005))
    fig.subplots_adjust(left=.18,right=.99,top=.91,bottom=.22)
    fig.savefig(out/'natural_modules.pdf',metadata={'CreationDate':None,'ModDate':None},dpi=300)
    fig.savefig(out/'natural_modules.png',dpi=180);plt.close(fig)
    (out/'natural_modules_binding.json').write_text(json.dumps(dict(source='results/natural-modules/evidence.zip',source_sha256=hashlib.sha256((ROOT/'results/natural-modules/evidence.zip').read_bytes()).hexdigest(),matrix=matrix,modules=modules,interpretation='Fixed attempt classifications, not independent-project counts or a population failure estimate'),indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);draw(p.parse_args().out)
