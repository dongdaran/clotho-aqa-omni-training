"""Fast deterministic learning-rate search using sampled source-question variants."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from baseline_utils import load_baseline
from experiment_utils import audit_data, read_rows, sample_stream, source_groups, stratified_subset, write_json, write_rows
from tune import execute_job, environment_for_gpu, run_attempt

LRS = (5e-5, 1e-4, 2.5e-5)
FIELDS = ['number','state','learning_rate','rank','microbatch','gradient_accumulation','max_steps',
          'best_step','sample_val_macro_f1','full_val_macro_f1','stop_reason','adapter']

def save_table(path, rows):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,delimiter='\t');w.writeheader();w.writerows(rows)
    tmp.replace(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    ap.add_argument('--output-dir',type=Path,default=Path('results/a100_macro_f1/search_greedy'))
    ap.add_argument('--baseline-metrics',type=Path,default=Path('results/a100_macro_f1/baseline/val_predictions.metrics.json'))
    ap.add_argument('--max-steps',type=int,default=150)
    ap.add_argument('--eval-steps',type=int,default=50)
    ap.add_argument('--min-steps',type=int,default=100)
    ap.add_argument('--val-size',type=int,default=200)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if not (args.max_steps >= args.min_steps >= args.eval_steps >= 1):
        ap.error('Invalid step budget')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    audit=audit_data(args.data_dir); write_json(args.output_dir/'data_audit.json',audit)
    baseline_full=load_baseline(args.baseline_metrics,args.data_dir/'val.jsonl')
    train=read_rows(args.data_dir/'train.jsonl'); groups=source_groups(train)
    schedule=sample_stream(groups,args.max_steps*64,args.seed)
    train_path=args.output_dir/'train_sampled.jsonl';write_rows(train_path,schedule)
    val=stratified_subset(read_rows(args.data_dir/'val.jsonl'),args.val_size,args.seed)
    val_path=args.output_dir/'val_subset.jsonl';write_rows(val_path,val)
    # Re-score the stored baseline predictions on the exact fixed subset.
    from tune import subset_baseline
    base_subset=subset_baseline(args.baseline_metrics,val)
    config={'method':'greedy_lr','learning_rates':LRS,'max_steps_per_candidate':args.max_steps,
            'eval_steps':args.eval_steps,'min_steps':args.min_steps,'rank':16,'microbatch':16,
            'gradient_accumulation':4,'effective_batch_size':64,'sampled_rows_per_candidate':len(schedule),
            'all_source_questions':len(groups),'baseline_full_macro_f1':baseline_full,'baseline':baseline_full,'seed':args.seed}
    config_path=args.output_dir/'search_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Search settings changed; use a new output directory')
    write_json(config_path,config)
    write_json(args.output_dir/'performance.json',{'selected':{'batch_size':16,'workers':2,'gradient_checkpointing':True}})
    env=environment_for_gpu(); rows=[]
    runtime={'workers':2,'gradient_checkpointing':True}
    for number,lr in enumerate(LRS):
        dest=args.output_dir/f'greedy_{number:02d}'
        if (dest/'completed.json').exists():
            result=json.loads((dest/'completed.json').read_text())
        else:
            job={'mode':'search','seed':args.seed,'rank':16,'learning_rate':lr,
                 'batch_size':16,'workers':runtime['workers'],'gradient_checkpointing':runtime['gradient_checkpointing'],
                 'train_path':str(train_path.resolve()),'val_path':str(val_path.resolve()),
                 'baseline':base_subset,'max_steps':args.max_steps,'eval_steps':args.eval_steps,
                 'min_steps':args.min_steps,'patience':2,'interactive':False,
                 'output_dir':str(dest.resolve())}
            result=execute_job(job,env)
        if result.get('oom'): raise RuntimeError(f'Greedy candidate LR={lr:g} OOM at fixed microbatch 16')
        best=result.get('best') or {}
        rows.append({'number':number,'state':'COMPLETE','learning_rate':lr,'rank':16,'microbatch':16,
                     'gradient_accumulation':4,'max_steps':result['completed_steps'],'best_step':result.get('best_step'),
                     'sample_val_macro_f1':best.get('macro_f1'),'full_val_macro_f1':'',
                     'stop_reason':result.get('stop_reason'),'adapter':result.get('adapter')})
        save_table(args.output_dir/'trials.tsv',rows)
        print(f"Greedy candidate {number+1}/{len(LRS)} LR={lr:g}: sample val macro-F1={best.get('macro_f1')}",flush=True)
    finalists=sorted(rows,key=lambda r:r['sample_val_macro_f1'] or -1,reverse=True)[:2]
    for row in finalists:
        dest=args.output_dir/f"greedy_{int(row['number']):02d}"/'full_val'
        cmd=[__import__('sys').executable,'-u',str(Path(__file__).with_name('swift_train.py')),
             '--adapter',row['adapter'],'--data-dir',str(args.data_dir.resolve()),
             '--output-dir',str(dest.resolve()),'--split','val']
        run_attempt(cmd,env,dest/'console.log').check_returncode()
        row['full_val_macro_f1']=json.loads((dest/'evaluation.json').read_text())['macro_f1']
        save_table(args.output_dir/'trials.tsv',rows)
    best=max(finalists,key=lambda r:r['full_val_macro_f1'])
    save_table(args.output_dir/'best.tsv',[best])
    write_json(args.output_dir/'current.json',{'state':'FINISHED','best':best})
    print(f"Selected greedy LR={best['learning_rate']}: full val macro-F1={best['full_val_macro_f1']:.4f}",flush=True)

if __name__=='__main__': main()
