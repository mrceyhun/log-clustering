# log-clustering

In "old-notebooks" one can see some of the first tests of the clusterlogs module. Some extra testing can be found here: https://cernbox.cern.ch/index.php/s/kGcKstDb1vHpcN2

In the "workflow" folder one can see my attempts of putting information from jupyter notebooks into continuous workflow.

By running the "workflow.py" script from lxplus machine the FTS error messages are taken from HDFS, grouped into clusters and then sent to MonIT.

User should have an access to the Analytix cluster in order to be able to extract the messages.
Before running the script from lxplus machine user should setup environment to use HADOOP with the following commands:
```bash
 source "/cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh"
 source "/cvmfs/sft.cern.ch/lcg/etc/hadoop-confext/hadoop-swan-setconf.sh" analytix
```

One issue that I have experienced was not being able to connect to MonIT receiving the following error:
```
Error: 'Connection' object has no attribute 'set_ssl'
```
In my case it could be solved by running this command before running the workflow.py:
```
export PYTHONPATH="$(python3 -m site --user-site):$PYTHONPATH"
```
