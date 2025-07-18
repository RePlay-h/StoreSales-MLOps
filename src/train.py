import pandas as pd
import yaml

def random_sample(input_path):
    df = pd.read_csv(input_path)

    ## Random sample
    sample_frac = 0.10
    data_sample = df.sample(frac=sample_frac, random_state=101).reset_index(drop=True)
    print(df.head())



if __name__ == '__main__':

    ## Get random data sample
    params = yaml.safe_load(open('params.yaml'))['evaluate']
    random_sample(params['input'])