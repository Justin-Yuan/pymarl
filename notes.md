PyMARL notes 

---
## Installation 

### docker 
docker setup is for Ubuntu, automatically install StarCraft env  (with binary and maps), sacred repo and torch

to run in docker, do (where $GPU=0)
```bash 
bash run.sh 0 python3 src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=2s3z
```

useful Docker commands 
- docker ps 
- docker image ls/rm
- docker container ls/rm/stop/kill
- docker container exec NAME CMD 

### virtual env 
could create conda env, install dependencies from `requirements.txt` and run `install_sc2.sh`, but it only gets smac/StarCraft env for ubuntu; for mac, manually download StarCraft env from <https://starcraft2.com/en-us/>, copy the maps to `Applications/StarCraft II`

to run in virtual env, do (in activated env) 
```bash 
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=2s3z
```

---

## Env 

### StarCrafet II 
reference: <https://github.com/oxwhirl/smac>





---




---
