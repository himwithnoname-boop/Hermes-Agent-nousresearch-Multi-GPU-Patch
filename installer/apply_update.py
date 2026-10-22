from pathlib import Path
import argparse,subprocess,json,hashlib,os,uuid,datetime
BASES={'5b147ab036d6d19131bff8d20701b5001eab844a':'v0.22.3', '4d04303818b61bf3aa09278af48356315515965b':'v0.22.4'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
def git(root,*args):return subprocess.run(['git','-C',str(root),*map(str,args)],capture_output=True,text=True,encoding='utf-8',errors='replace')
def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['check','apply','complete']);a.add_argument('root');v=a.parse_args();root=Path(v.root).resolve();kit=Path(__file__).resolve().parent
 if not (root/'hermes_cli/main.py').is_file():raise RuntimeError('Choose the hermes-agent source folder, not its parent.')
 if root==kit or root in kit.parents:raise RuntimeError('Extract the installer outside the Hermes source folder.')
 for name,digest in json.loads((kit/'checksums.json').read_text(encoding='utf-8')).items():
  if sha(kit/name)!=digest:raise RuntimeError('Package integrity failure: '+name)
 head=git(root,'rev-parse','HEAD')
 if head.returncode or head.stdout.strip() not in BASES:raise RuntimeError('This installer requires an exact supported Hermes v0.22.3 or v0.22.4 checkout. No files changed.')
 version=BASES[head.stdout.strip()]
 cache=Path(os.environ['LOCALAPPDATA'])/'Hermes-MultiGPU-Patch'/hashlib.sha256(str(root).lower().encode()).hexdigest()[:16]/version
 receipt=cache/'receipt.json'
 if receipt.exists():
  r=json.loads(receipt.read_text(encoding='utf-8'))
  if r['sourceRoot']!=str(root):raise RuntimeError('Backup belongs to a different source folder.')
  if 'after' not in r:
   raise RuntimeError('An interrupted backup-only installation requires manual recovery from '+str(cache))
  if all(sha(root/p)==h for p,h in r['after'].items()):
   if v.action=='complete':r['status']='complete';receipt.write_text(json.dumps(r,indent=2),encoding='utf-8')
   print('Source fix already applied; backups: '+str(cache));return
  raise RuntimeError('Files changed since the previous installation. Preserve the backup and resolve manually: '+str(cache))
 selected=None
 for name in ('multigpu-v0.22.3.patch',):
  if git(root,'apply','--check','--',kit/name).returncode==0:selected=kit/name;break
 if selected is None:raise RuntimeError('The source does not match the supported stock Hermes v0.22.3/v0.22.4 baseline. No files changed.')
 paths=[line.split(' b/',1)[1] for line in selected.read_text(encoding='utf-8').splitlines() if line.startswith('diff --git a/')]
 for p in paths:
  target=root/p
  if not target.resolve().is_relative_to(root) or target.is_symlink():raise RuntimeError('Unsafe source path: '+p)
 print('Compatible: '+selected.name+'; '+str(len(paths))+' source files.')
 if v.action=='check':return
 if v.action=='complete':raise RuntimeError('No source-install receipt exists.')
 cache.mkdir(parents=True,exist_ok=False)
 before={}
 for p in paths:
  src=root/p;before[p]=sha(src)
  if src.exists():dest=cache/'original'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
 r={'schemaVersion':1,'sourceRoot':str(root),'status':'backed-up','patch':selected.name,'before':before,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 receipt.write_text(json.dumps(r,indent=2),encoding='utf-8')
 result=git(root,'apply','--',selected)
 if result.returncode:raise RuntimeError('Source apply failed; backups retained. '+result.stderr)
 r.update(status='source-applied-desktop-build-required',after={p:sha(root/p) for p in paths});receipt.write_text(json.dumps(r,indent=2),encoding='utf-8')
 print('Source applied. Backups and receipt: '+str(cache))
if __name__=='__main__':
 try:main()
 except Exception as e:raise SystemExit(str(e))
