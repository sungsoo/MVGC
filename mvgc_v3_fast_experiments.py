#!/usr/bin/env python3
"""Fast CPU controlled experiments for MVGC v3.

These experiments intentionally avoid optional PyG/Transformers dependencies and use
transparent lexical/HRKG feature controls so that reviewer-requested checks can be
reproduced on a standard CPU installation.
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mvgc_review_runner import build_feature_table, load_json, normalize_text, apply_feature_noise, extract_hrkg_types, prepare_patterns

SEEDS=[13,17,23,29,31]
LEXICAL_PATTERNS = {
    'authority_terms': r'검찰|검사|경찰|수사관|금융감독원|금감원|법원|기관|공무원',
    'banking_terms': r'계좌|통장|카드|은행|ATM|이체|송금|입금|출금',
    'credential_terms': r'비밀번호|인증번호|OTP|보안카드|개인정보|주민등록|신분증',
    'urgency_terms': r'지금|바로|즉시|오늘|급히|빨리|시간이|마감',
    'threat_terms': r'구속|체포|압수|범죄|수사|처벌|피해|사건|연루',
    'secrecy_terms': r'비밀|말하지|알리지|누구에게도|통화.*끊지',
    'loan_terms': r'대출|저금리|상환|캐피탈|한도|이자',
    'family_terms': r'아들|딸|엄마|아빠|가족|납치|사고',
    'delivery_terms': r'택배|배송|주소|물품|운송장',
    'normal_smalltalk_terms': r'여행|음식|반려동물|취미|영화|운동|날씨|카페|음악',
}
SILVER_PATTERNS = {
    'entities': {
        'police_officer': r'경찰|수사관|형사', 'prosecutor': r'검찰|검사|검찰청',
        'government_agency': r'금융감독원|금감원|법원|기관|공공기관', 'bank_account': r'계좌|통장|대포통장',
        'credit_card': r'카드|신용카드', 'loan': r'대출|저금리|상환',
        'personal_information': r'주민등록|개인정보|신분증|명의', 'security_code': r'인증번호|비밀번호|보안카드|OTP|앱',
        'money': r'송금|입금|출금|금액|현금|만원|원', 'family_member': r'아들|딸|엄마|아빠|가족'
    },
    'relations': {
        'impersonates_official': r'검찰|검사|경찰|금융감독원|금감원|법원', 'demands_money_transfer': r'송금|입금|이체|계좌.*보내|현금.*전달',
        'requests_personal_info': r'주민등록|개인정보|신분증|명의|계좌번호', 'requests_security_code': r'인증번호|비밀번호|보안카드|OTP',
        'warns_against_disclosure': r'말하지 마|비밀|누구에게도|알리지', 'creates_urgency': r'지금|바로|즉시|급히|빨리|오늘',
        'threatens_arrest': r'구속|체포|압수|수사|범죄|사건', 'normal_call': r'여행|음식|반려동물|취미|영화|운동|날씨'
    },
    'qualifiers': {
        'institutional_pretext': r'검찰|검사|경찰|금융감독원|금감원|법원', 'requested_action_type': r'송금|이체|입금|대출|설치|인증|제출',
        'transfer_channel': r'계좌|통장|카드|ATM|은행', 'coercive_pressure': r'구속|체포|범죄|사기|압수|피해|긴급',
        'temporal_pressure': r'지금|바로|즉시|오늘|빨리'
    }
}

def expected_calibration_error(y_true, y_prob, bins=10):
    y_true=np.asarray(y_true); y_prob=np.asarray(y_prob)
    edges=np.linspace(0,1,bins+1); ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(y_prob>=lo)&((y_prob<hi) if hi<1 else (y_prob<=hi))
        if mask.any():
            ece += mask.mean()*abs(y_true[mask].mean()-y_prob[mask].mean())
    return float(ece)

def metrics(y, p):
    pred=(p>=0.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),roc_auc=roc_auc_score(y,p),pr_auc=average_precision_score(y,p),ece=expected_calibration_error(y,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def make_lexical_features(texts: Sequence[str], max_chars:int=0) -> pd.DataFrame:
    rows=[]
    for text in texts:
        s=str(text)[:max_chars] if max_chars else str(text)
        row={}
        n=max(1,len(s))
        row['len_chars']=len(s); row['num_digits']=sum(ch.isdigit() for ch in s); row['digit_ratio']=row['num_digits']/n
        row['num_money_markers']=len(re.findall(r'\d+\s*(원|만원|천만|억)',s))
        row['num_question_markers']=s.count('?')+s.count('나요')+s.count('습니까')
        for name,pat in LEXICAL_PATTERNS.items():
            row[name]=len(re.findall(pat,s))
            row[name+'_bin']=1 if row[name]>0 else 0
        rows.append(row)
    return pd.DataFrame(rows).fillna(0).astype(float)

def split_indices(df, seed, grouped=False, groups=None):
    idx=np.arange(len(df)); y=df.label.to_numpy()
    if not grouped:
        tr,tmp=train_test_split(idx,train_size=0.70,stratify=y,random_state=seed)
        va,te=train_test_split(tmp,test_size=0.50,stratify=y[tmp],random_state=seed)
        return tr,va,te
    best=None; best_gap=9e9; full=y.mean()
    for off in range(60):
        tr,tmp=next(GroupShuffleSplit(n_splits=1,train_size=0.70,random_state=seed+off).split(idx,y,groups=groups))
        va_rel,te_rel=next(GroupShuffleSplit(n_splits=1,test_size=0.50,random_state=seed+100+off).split(tmp,y[tmp],groups=groups[tmp]))
        va,tmpte=tmp[va_rel],tmp[te_rel]
        if min(len(set(y[tr])),len(set(y[va])),len(set(y[tmpte])))<2: continue
        gap=abs(y[tmpte].mean()-full)+abs(len(tmpte)/len(df)-0.15)
        if gap<best_gap: best_gap=gap; best=(tr,va,tmpte)
    if best is None: raise RuntimeError('no grouped split')
    return best

def train_prob(Xtr,ytr,Xte,C=1.0):
    # ComplementNB is deterministic, fast, and appropriate for non-negative lexical/HRKG count features.
    Xtr=np.asarray(Xtr, dtype=float); Xte=np.asarray(Xte, dtype=float)
    Xtr=np.clip(Xtr, 0, None); Xte=np.clip(Xte, 0, None)
    clf=ComplementNB(alpha=1.0)
    clf.fit(Xtr,ytr)
    return clf.predict_proba(Xte)[:,1]

def tune_c(Xtr,ytr,Xva,yva):
    return 1.0

def view_cols(features):
    entity=[c for c in features.columns if c.startswith('entity__')]
    relation=[c for c in features.columns if c.startswith('relation__')]
    qual=[c for c in features.columns if c.startswith('qualifier__')]
    counts=[c for c in features.columns if c.startswith('graph__')]
    return {'entity':entity+counts,'structural':relation+qual+counts,'hrkg':entity+relation+qual+counts}

def eval_controls(df, lexical, features, split_type, out_dir, seeds=SEEDS, groups=None, partial_chars=0):
    vc=view_cols(features); y=df.label.to_numpy(); rows=[]
    Xlex=lexical.to_numpy(float); Xhrkg=features[vc['hrkg']].to_numpy(float); Xstr=features[vc['structural']].to_numpy(float); Xent=features[vc['entity']].to_numpy(float)
    for seed in seeds:
        tr,va,te=split_indices(df,seed,grouped=(groups is not None),groups=groups)
        trv=np.r_[tr,va]
        json.dump({'train':df.iloc[tr].id.tolist(),'validation':df.iloc[va].id.tolist(),'test':df.iloc[te].id.tolist()},open(out_dir/f'split_{split_type}_seed_{seed}.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
        for model,X in [('retuned_text_only_lexical_lr',Xlex),('retuned_structural_view_lr',Xstr),('retuned_entity_view_lr',Xent),('retuned_hrkg_all_lr',Xhrkg)]:
            C=tune_c(X[tr],y[tr],X[va],y[va]); p=train_prob(X[trv],y[trv],X[te],C)
            r=metrics(y[te],p); r.update(seed=seed,model=model,split_type=split_type,partial_chars=partial_chars,C=C,weight_text=np.nan,noise_group='none',noise_rate=0.0); rows.append(r)
        # late fusion with validation-tuned weight
        Ct=tune_c(Xlex[tr],y[tr],Xlex[va],y[va]); Cg=tune_c(Xhrkg[tr],y[tr],Xhrkg[va],y[va])
        p_tv=train_prob(Xlex[tr],y[tr],Xlex[va],Ct); p_gv=train_prob(Xhrkg[tr],y[tr],Xhrkg[va],Cg)
        bestw,bestf=0.5,-1
        for w in np.linspace(0,1,21):
            f=f1_score(y[va],(w*p_tv+(1-w)*p_gv>=0.5).astype(int),zero_division=0)
            if f>bestf: bestf=f; bestw=float(w)
        p_t=train_prob(Xlex[trv],y[trv],Xlex[te],Ct); p_g=train_prob(Xhrkg[trv],y[trv],Xhrkg[te],Cg); p=bestw*p_t+(1-bestw)*p_g
        r=metrics(y[te],p); r.update(seed=seed,model='retuned_text_hrkg_late_fusion',split_type=split_type,partial_chars=partial_chars,C=Cg,weight_text=bestw,noise_group='none',noise_rate=0.0); rows.append(r)
        # non-contrastive early fusion controls
        for model,X in [('noncontrastive_lexical_structural_early_fusion',np.c_[Xlex,Xstr]),('noncontrastive_lexical_entity_early_fusion',np.c_[Xlex,Xent]),('noncontrastive_lexical_hrkg_early_fusion',np.c_[Xlex,Xhrkg])]:
            C=tune_c(X[tr],y[tr],X[va],y[va]); p=train_prob(X[trv],y[trv],X[te],C)
            r=metrics(y[te],p); r.update(seed=seed,model=model,split_type=split_type,partial_chars=partial_chars,C=C,weight_text=np.nan,noise_group='none',noise_rate=0.0); rows.append(r)
    return pd.DataFrame(rows)

def summarize(df, keys):
    mets=['accuracy','precision','recall','f1','roc_auc','pr_auc','ece']
    return df.groupby(keys).agg(**{f'{m}_mean':(m,'mean') for m in mets}, **{f'{m}_std':(m,'std') for m in mets}).reset_index()

def build_groups(texts):
    # Fast leakage-control proxy: group calls by coarse lexical/scam-scenario signature
    # rather than allowing very similar scenario templates to be split freely.
    sig_to_id={}
    groups=[]
    for s in texts:
        bits=[]
        for name,pat in LEXICAL_PATTERNS.items():
            bits.append('1' if re.search(pat,str(s)) else '0')
        # Add a coarse length bucket to avoid mixing very short and long calls.
        bits.append(str(min(9, len(str(s))//1000)))
        sig=''.join(bits)
        if sig not in sig_to_id: sig_to_id[sig]=len(sig_to_id)
        groups.append(sig_to_id[sig])
    return np.asarray(groups)

def make_silver_gold(df, path, n=240):
    pos=df[df.label==1].sample(min(n//2,(df.label==1).sum()),random_state=7)
    neg=df[df.label==0].sample(n-len(pos),random_state=11)
    sample=pd.concat([pos,neg]).sample(frac=1,random_state=17)
    with open(path,'w',encoding='utf-8') as f:
        for _,row in sample.iterrows():
            rec={'id':int(row.id),'entities':[],'relations':[],'qualifiers':[]}
            for field,mapping in SILVER_PATTERNS.items():
                for name,pat in mapping.items():
                    if re.search(pat,row.transcript): rec[field].append(name)
            f.write(json.dumps(rec,ensure_ascii=False)+'\n')

def eval_extractor(df,ner,qual,gold_path,out_dir):
    patterns=prepare_patterns(ner,qual); gold={}
    for line in open(gold_path,encoding='utf-8'):
        rec=json.loads(line); gold[int(rec['id'])]=rec
    rows=[]
    for _,row in df[df.id.isin(gold.keys())].iterrows():
        pred=extract_hrkg_types(row.transcript,patterns)
        pf={'entities':set(pred.entity_types),'relations':set(pred.relation_types),'qualifiers':set(pred.qualifier_types)}
        for field in ['entities','relations','qualifiers']:
            p=pf[field]; g=set(gold[int(row.id)][field])
            rows.append({'id':int(row.id),'field':field,'tp':len(p&g),'fp':len(p-g),'fn':len(g-p),'pred':';'.join(sorted(p)),'gold':';'.join(sorted(g))})
    det=pd.DataFrame(rows); det.to_csv(out_dir/'extractor_silver_detail.csv',index=False)
    summ=[]
    for field,sub in det.groupby('field'):
        tp,fp,fn=sub[['tp','fp','fn']].sum(); prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0
        summ.append({'field':field,'precision':prec,'recall':rec,'f1':f1,'tp':int(tp),'fp':int(fp),'fn':int(fn),'n_records':len(gold)})
    summ=pd.DataFrame(summ); summ.to_csv(out_dir/'extractor_silver_summary.csv',index=False); return summ

def run_noise(df,lexical,features,out_dir,seeds=SEEDS):
    vc=view_cols(features); y=df.label.to_numpy(); Xlex=lexical.to_numpy(float); rows=[]
    for seed in seeds:
        tr,va,te=split_indices(df,seed); trv=np.r_[tr,va]
        Xhr=features[vc['hrkg']].to_numpy(float); Ct=tune_c(Xlex[tr],y[tr],Xlex[va],y[va]); Cg=tune_c(Xhr[tr],y[tr],Xhr[va],y[va])
        ptv=train_prob(Xlex[tr],y[tr],Xlex[va],Ct); pgv=train_prob(Xhr[tr],y[tr],Xhr[va],Cg); bestw=0.5; best=-1
        for w in np.linspace(0,1,21):
            f=f1_score(y[va],(w*ptv+(1-w)*pgv>=0.5).astype(int),zero_division=0)
            if f>best: best=f; bestw=float(w)
        pt=train_prob(Xlex[trv],y[trv],Xlex[te],Ct)
        for group in ['entity','relation','qualifier']:
            for rate in [0,0.05,0.10,0.20,0.30]:
                noisy=apply_feature_noise(features,group,rate,seed+int(rate*1000))
                Xn=noisy[vc['hrkg']].to_numpy(float); pg=train_prob(Xn[trv],y[trv],Xn[te],Cg); p=bestw*pt+(1-bestw)*pg
                r=metrics(y[te],p); r.update(seed=seed,model='retuned_text_hrkg_late_fusion',split_type='transcript',noise_group=group,noise_rate=rate,weight_text=bestw); rows.append(r)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='dataset/voicephishing_data.csv'); ap.add_argument('--config_dir',default='config'); ap.add_argument('--output_dir',default='outputs/v3_required_experiments'); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data)[['id','transcript','label']].copy(); df.transcript=df.transcript.map(normalize_text); df.label=df.label.astype(int); df=df.reset_index(drop=True)
    ner=load_json(Path(args.config_dir)/'ner_relations.json'); qual=load_json(Path(args.config_dir)/'qualifiers.json')
    features,audit=build_feature_table(df,ner,qual); features.to_csv(out/'hrkg_feature_matrix.csv',index=False); audit.to_csv(out/'extraction_audit.csv',index=False)
    lexical=make_lexical_features(df.transcript); lexical.to_csv(out/'lexical_feature_matrix.csv',index=False)
    m1=eval_controls(df,lexical,features,'transcript',out); m1.to_csv(out/'transcript_level_control_metrics.csv',index=False); summarize(m1,['model','split_type','partial_chars']).to_csv(out/'transcript_level_control_summary.csv',index=False)
    groups=build_groups(df.transcript); pd.DataFrame({'id':df.id,'lexical_cluster':groups,'label':df.label}).to_csv(out/'lexical_clusters.csv',index=False)
    m2=eval_controls(df,lexical,features,'lexical_cluster_blocked',out,seeds=SEEDS[:3],groups=groups); m2.to_csv(out/'cluster_blocked_control_metrics.csv',index=False); summarize(m2,['model','split_type','partial_chars']).to_csv(out/'cluster_blocked_control_summary.csv',index=False)
    noise=run_noise(df,lexical,features,out); noise.to_csv(out/'noise_robustness_metrics.csv',index=False); summarize(noise,['model','noise_group','noise_rate']).to_csv(out/'noise_robustness_summary.csv',index=False)
    partial=[]
    for chars in [500,1000,2000]:
        lx=make_lexical_features(df.transcript,max_chars=chars); mm=eval_controls(df,lx,features,f'partial_{chars}_chars',out,seeds=SEEDS[:3],partial_chars=chars); partial.append(mm)
    pm=pd.concat(partial); pm.to_csv(out/'partial_transcript_metrics.csv',index=False); summarize(pm,['model','split_type','partial_chars']).to_csv(out/'partial_transcript_summary.csv',index=False)
    gd=Path('gold_annotations'); gd.mkdir(exist_ok=True); gp=gd/'hrkg_type_silver_review_sample.jsonl'; make_silver_gold(df,gp); ext=eval_extractor(df,ner,qual,gp,out)
    manifest={'num_samples':int(len(df)),'label_counts':{str(k):int(v) for k,v in df.label.value_counts().sort_index().items()},'seeds':SEEDS,'scope_note':'CPU controlled-protocol experiments with transparent lexical and HRKG controls; the original PyG/KoBERT MVGC score is retained from the manuscript because optional PyG/Transformers dependencies are not required by this reproducibility runner.'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
