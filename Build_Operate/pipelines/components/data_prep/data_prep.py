import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Parse job parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--test_size', type=float, default=0.30)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--test_data', type=str)
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    data = pd.read_csv(args.data, header=0)
    print("num_samples:", data.shape[0])
    print("num_features", data.shape[1] - 1)

    train_data, test_data,  = train_test_split(data, test_size=args.test_size, random_state=0)

    # args are mounted folders, so write the data into the folder
    train_data.to_csv(os.path.join(args.train_data,"train_data.csv"))
    test_data.to_csv(os.path.join(args.test_data,"test_data.csv"))
    
if __name__ == "__main__":
    main()
