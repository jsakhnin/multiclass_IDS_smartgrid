import sys
import argparse
from typing import Optional, Sequence

import experiments

commands = {
    "basic-implementation": experiments.basic_implementation,
    "features-heatmap": experiments.features_heatmap,
    "pearson": experiments.pearson_correlations,
    "pearson-split": experiments.pearson_correlations_split,
    "cv": experiments.cv_experiment,
    "cm": experiments.cmatrix,
    "scalability": experiments.scalability_test,
}

def main(argv: Optional[Sequence[str]] = None ) -> int:
    print('\n\n----------------------------------\n  exec  \n--\n\n')
    parser = argparse.ArgumentParser()

    # ------------------------------- Sub commands
    subparser = parser.add_subparsers(dest='command', help = 'Executive Commands')
    subparser.required = True
 
    # Basic implemetaion
    basic_implementation_parser = subparser.add_parser('basic-implementation', help="Basic implementation of the ERLC model")
    # Features heatmap
    fheatmap_parser = subparser.add_parser('features-heatmap', help="Produces features heatmap results")
    # Pearson correlation
    pcorr_parser = subparser.add_parser('pearson', help="Pearson correlation test for localizer")
    # Pearson correlation split
    pcorr_split_parser = subparser.add_parser('pearson-split', help="Pearson correlation split test for localizer")
    # Cross validation
    cv_parser = subparser.add_parser('cv', help="Cross validation of attack classfication of all models")
    # Confusion matrix
    cm_parser = subparser.add_parser('cm', help="Confusion matrix for erlc")

    # Scalability
    scalability_parser = subparser.add_parser('scalability', help="Scalability results")


    ## Parse arguments
    args = parser.parse_args(argv)

    user_args = vars(args)
    user_command = user_args.pop('command')
    commands[user_command](**user_args)
    return 0


if __name__ == '__main__':
    exit(main())