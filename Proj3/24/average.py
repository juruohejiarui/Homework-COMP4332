import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("input", type=str, nargs='+')
parser.add_argument("--output", type=str, default="prediction.csv")

if __name__ == "__main__" :
	args = parser.parse_args()

	datas : list[pd.DataFrame] = [pd.read_csv(input) for input in args.input]

	out = datas[0][["ReviewerID", "ProductID"]].copy()

	input_cols = []
	for i, data in enumerate(datas) :
		out[f'input_{i}'] = data['Star'].copy()
		input_cols.append(f'input_{i}')
	
	out['Star'] = out[input_cols].mean(axis=1)

	print(out.head(5))

	out[["ReviewerID", "ProductID", "Star"]].to_csv(args.output, index=False)