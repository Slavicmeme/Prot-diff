import pandas as pd
import argparse

def filter_grampa_by_bacterium(input_csv, output_csv, bacterium_keyword="E. coli", max_len=50, mic_threshold=1.2):
    df = pd.read_csv(input_csv)
    df = df[df['bacterium'].str.contains(bacterium_keyword, na=False)]
    df = df[df['value'] < mic_threshold]
    df = df.drop_duplicates(subset='sequence')
    df = df[df['sequence'].str.len() <= max_len]
    df['sequence'] = df['sequence'].str.replace(r'[^ACDEFGHIKLMNPQRSTVWY]', '', regex=True)
    df[['sequence', 'value']].to_csv(output_csv, index=False)
    print(f"Saved {len(df)} sequences to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to grampa.csv")
    parser.add_argument("--output_csv", type=str, required=True, help="Filtered output CSV path")
    parser.add_argument("--bacterium", type=str, default="E. coli", help="Bacterium keyword filter")
    args = parser.parse_args()
    filter_grampa_by_bacterium(args.input_csv, args.output_csv, bacterium_keyword=args.bacterium)
