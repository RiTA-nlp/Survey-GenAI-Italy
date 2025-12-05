import pandas as pd
import argparse

def main(v2_file, v3_file, output_file):
    # Read TSVs using only the first row as header
    df_v2 = pd.read_csv(v2_file, sep="\t", header=0)
    df_v3 = pd.read_csv(v3_file, sep="\t", header=0)

    # Merge Q26_mapped from v2 into v3 using ResponseId
    df_v3 = df_v3.merge(
        df_v2[["ResponseId", "Q26_mapped"]],
        on="ResponseId",
        how="left"
    )

    # Rename merged column
    df_v3.rename(columns={"Q26_mapped": "mapped_job"}, inplace=True)

    # Save
    df_v3.to_csv(output_file, sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map Q26_mapped from v2 to v3 TSV by ResponseId")
    parser.add_argument("v2_file", help="Path to survey_v2_clean.tsv")
    parser.add_argument("v3_file", help="Path to survey_v3_clean.tsv")
    parser.add_argument("output_file", help="Path to save survey_v3_with_mapped_job.tsv")

    args = parser.parse_args()
    main(args.v2_file, args.v3_file, args.output_file)

