"""Getting params from the command line."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run GedGNN.")

    parser.add_argument("--epochs",
                        type=int,
                        default=1,
	                help="Number of training epochs. Default is 1.")

    parser.add_argument("--filters-1",
                        type=int,
                        default=128,
	                help="Filters (neurons) in 1st convolution. Default is 64.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=64,
	                help="Filters (neurons) in 2nd convolution. Default is 32.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=32,
	                help="Filters (neurons) in 3rd convolution. Default is 16.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
	                help="Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
	                help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--bottle-neck-neurons-2",
                        type=int,
                        default=8,
                        help="2nd bottle neck layer neurons. Default is 8.")

    parser.add_argument("--bottle-neck-neurons-3",
                        type=int,
                        default=4,
                        help="3rd bottle neck layer neurons. Default is 4.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
	                help="Similarity score bins. Default is 16.")

    parser.add_argument("--hidden-dim",
                        type=int,
                        default=16,
                        help="the size of weight matrix in GedMatrixModule. Default is 16.")

    parser.add_argument("--histogram",
                        dest="histogram",
                        default=False,
                        help='Whether to use histogram.')

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                help="Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--demo",
                        dest="demo",
                        action="store_true",
                        default=False,
                        help='Generate just a few graph pairs for training and testing.')

    parser.add_argument("--gtmap",
                        dest="gtmap",
                        action="store_true",
                        default=False,
                        help='Whether to pack gt mapping')

    parser.add_argument("--finetune",
                        dest="finetune",
                        action="store_true",
                        default=False,
                        help='Whether to use finetune.')

    parser.add_argument("--prediction-analysis",
                        action="store_true",
                        default=False,
                        help='Whether to analyze the bias of prediction.')

    parser.add_argument("--postk",
                        type=int,
                        default=100,
                        help="Find k-best matching in the post-processing algorithm. Default is 1000.")

    parser.add_argument("--abs-path",
                        type=str,
                        default="",
                        # default='/apdcephfs/private_czpiao/workplace/gedgnn/',
                        help="the absolute path")

    parser.add_argument("--result-path",
                        type=str,
                        default='result/',
                        help="Where to save the evaluation results")

    parser.add_argument("--model-train",
                        type=int,
                        default=1,
                        help='Whether to train the model')

    parser.add_argument("--model-path",
                        type=str,
                        default='model_save/',
                        help="Where to save the trained model")

    parser.add_argument("--model-epoch-start",
                        type=int,
                        default=0,
                        help="The number of epochs the initial saved model has been trained.")

    parser.add_argument("--model-epoch-end",
                        type=int,
                        default=0,
                        help="The number of epochs the final saved model has been trained.")

    parser.add_argument("--dataset",
                        type=str,
                        default='AIDS',
                        help="dataset name")

    parser.add_argument("--model-name",
                        type=str,
                        default='GPN',
                        help="model name, including [GPN, SimGNN, TaGSim, GedGNN].")

    parser.add_argument("--graph-pair-mode",
                        type=str,
                        default='combine',
                        help="The way of generating graph pairs, including [normal, delta, combine].")

    parser.add_argument("--target-mode",
                        type=str,
                        default='linear',
                        help="The way of generating target, including [linear, exp].")

    parser.add_argument("--greedy",
                        dest="greedy",
                        action="store_true",
                        default=False,
                        help='Whether to use greedy matching matrix (Hungarian).')

    parser.add_argument("--num-delta-graphs",
                        type=int,
                        default=100,
                        help="The number of synthetic delta graph pairs for each graph.")

    parser.add_argument("--num-testing-graphs",
                        type=int,
                        default=100,
                        help="The number of testing graph pairs for each graph.")

    parser.add_argument("--num-labels",
                        type=int,
                        default=10,
                        help="The number of node labels.")

    parser.add_argument("--loss-weight",
                        type=float,
                        default=1.0,
                        help="In GedGNN, the weight of value loss. Default is 1.0.")

    parser.add_argument("--delta-graph-seed",
                        type=int,
                        default=0,
                        help="The random seed used for generating delta graphs.")

    return parser.parse_args()
