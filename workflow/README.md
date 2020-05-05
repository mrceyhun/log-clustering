## How to run workflow.py

I will be assuming that the user has access to the Analytix cluster.

After cloning the repo to lxplus go to the workflow directory:
```
cd log-clustering/workflow/
```

In a new virtual environment install the libraries from requirements file:
```
pip install -r requirements.txt
```
Also, `clusterlogs` folder must stay in the same directory as `workflow.py`.

Setup environment to use HADOOP with the following commands:
```bash
 source "/cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh"
 source "/cvmfs/sft.cern.ch/lcg/etc/hadoop-confext/hadoop-swan-setconf.sh" analytix
```

Now you can run workflow.py

## Possible issues

One issue that I have experienced was not being able to connect to MonIT receiving the following error:
```
Error: 'Connection' object has no attribute 'set_ssl'
```
In my case it could be solved by running this command after the `source "/cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh"` command:
```
export PYTHONPATH="$(python3 -m site --user-site):$PYTHONPATH"
```

