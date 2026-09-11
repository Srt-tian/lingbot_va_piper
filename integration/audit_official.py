"""Compare executed core functions against a pinned official checkout, without imports."""
import argparse,ast,hashlib,json
from pathlib import Path

def functions(path):
    result={}
    def walk(nodes,prefix=''):
        for node in nodes:
            if isinstance(node,ast.ClassDef):walk(node.body,prefix+node.name+'.')
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                result[prefix+node.name]=ast.dump(node,include_attributes=False)
    walk(ast.parse(path.read_text()).body);return result

def compare(official,ours):
    required={
      'wan_va/wan_va_server.py':['VA_Server.'+s for s in ['_infer','_compute_kv_cache','_encode_obs','_reset','_prepare_latent_input','preprocess_action','postprocess_action','infer']],
      'wan_va/train.py':['Trainer.'+s for s in []],
      'wan_va/modules/utils.py':[],
      'wan_va/modules/model.py':[],
      'wan_va/utils/scheduler.py':[],
      'wan_va/dataset/lerobot_latent_dataset.py':[]}
    result={};failures=[]
    for file,critical in required.items():
        left,right=official/file,ours/file;a,b=functions(left),functions(right)
        changed=[k for k in a.keys()&b.keys() if a[k]!=b[k]]
        # Whitelist mathematical training functions by name, independent of class naming.
        if file=='wan_va/train.py':critical=[k for k in a if k.split('.')[-1] in ('_add_noise','_prepare_input_dict','compute_loss')]
        if file=='wan_va/modules/model.py':critical=[k for k in a if k.split('.')[-1] in ('forward','forward_train','_get_mask_mod','_get_cross_mask_mod','allocate_slots','update_cache','restore_cache')]
        bad=[k for k in critical if a.get(k)!=b.get(k)]
        failures.extend(file+':'+k for k in bad)
        result[file]={'official_sha256':hashlib.sha256(left.read_bytes()).hexdigest(),
            'ours_sha256':hashlib.sha256(right.read_bytes()).hexdigest(),
            'changed_functions':sorted(changed),'added_functions':sorted(b.keys()-a.keys()),
            'critical_functions_checked':sorted(critical),'critical_mismatches':bad}
    return {'core_pass':not failures,'failures':failures,'files':result}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--official',type=Path,required=True)
    parser.add_argument('--ours',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=compare(args.official,args.ours)
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['core_pass'] else 1)
