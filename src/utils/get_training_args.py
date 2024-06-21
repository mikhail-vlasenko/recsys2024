import argparse
from src.data.ebnerd_variants import EbnerdVariants

def get_training_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ebnerd_variant",
        type=str,
        choices=EbnerdVariants.get_variants(),
        default=EbnerdVariants.get_default_variant(),
    )

    parser.add_argument("--num_workers", type=int, default=0)

    #hugging face
    parser.add_argument("--api_key", type=str, default=None, help="Hugging Face API key")

    #graph method specific arguments 
    parser.add_argument('--dataset', type=str, default='ten_week', help='which dataset to use')
    parser.add_argument('--title_len', type=int, default=10, help='the max length of title')
    parser.add_argument('--session_len', type=int, default=10, help='the max length of session')
    parser.add_argument('--aggregator', type=str, default='neighbor', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=1, help='the number of epochs')
    parser.add_argument('--user_neighbor', type=int, default=30, help='the number of neighbors to be sampled')
    parser.add_argument('--news_neighbor', type=int, default=10, help='the number of neighbors to be sampled')
    parser.add_argument('--entity_neighbor', type=int, default=1, help='the number of neighbors to be sampled') #whats this one
    parser.add_argument('--user_dim', type=int, default=128, help='dimension of user and entity embeddings')
    parser.add_argument('--cnn_out_size', type=int, default=128, help='dimension of cnn output')
    parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=5e-3, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')  #3e-4
    parser.add_argument('--save_path', type=str, default="./data/1week/hop2/version1/", help='model save path')
    parser.add_argument('--test', type=int, default=0, help='test')
    parser.add_argument('--use_group', type=int, default=1, help='whether use group')
    parser.add_argument('--n_filters', type=int, default=64, help='number of filters for each size in KCNN')
    parser.add_argument('--filter_sizes', type=int, default=[2, 3], nargs='+',
                        help='list of filter sizes, e.g., --filter_sizes 2 3')
    parser.add_argument('--ncaps', type=int, default=7,
                        help='Maximum number of capsules per layer.')
    parser.add_argument('--dcaps', type=int, default=0, help='Decrease this number of capsules per layer.')
    parser.add_argument('--nhidden', type=int, default=16,
                            help='Number of hidden units per capsule.')
    parser.add_argument('--routit', type=int, default=7,
                        help='Number of iterations when routing.')
    parser.add_argument('--balance', type=float, default=0.004, help='learning rate')  #3e-4
    parser.add_argument('--version', type=int, default=0,
                            help='Different version under the same set')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--optimized_subsampling', action='store_true')
    parser.set_defaults(optimized_subsampling=False)

    parser.add_argument('--more_labels', action='store_false', default=True)
    parser.add_argument('--history_size', type=int, default=30, help=' The maximum size of the history to retain')
    parser.add_argument('--fraction', type=float, default=1.0, help='fraction of data to use for the behaviors df, number applies to both train, test and val')
    parser.add_argument('--npratio', type=int, default=4, help='The ratio of negative article ids to positive article ids in train and val data')
    parser.add_argument('--one_row_impression', action='store_true', default=False)
    
    show_loss = True
    show_time = False

    #t = time()

    args = parser.parse_args()
    if args.api_key is None:
        with open("api_key.txt") as f:
            args.api_key = f.read().strip()
    return args