import pandas as pd

for i in ('0101','0102','0115','0201','0225'):
    df = pd.read_csv('error'+i+'.txt').head(100000)
    df.to_csv('errorSHORT'+i+'.txt',index=False, header=True)
