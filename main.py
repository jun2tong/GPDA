import argparse
import torch
import os

from src.solver import Solver

###############################################################################

#
# hyperparameters
#

parser = argparse.ArgumentParser()

parser.add_argument('--nsamps_q',
                    type=int, default=50,
                    help='# of samples from variational density q(w) (default: 50)')

parser.add_argument('--lamb_marg_loss',
                    type=float, default=50.0,
                    help='impact of margin loss (default: 10.0)')

parser.add_argument('--all_use',
                    type=str, default='no',
                    help='use all training data? (default: "no")')

parser.add_argument('--batch-size',
                    type=int, default=32,
                    help='input batch size for training (default: 128)')

# parser.add_argument( '--checkpoint_dir',
#  type=str, default='checkpoint',
#  help='source only or not (default: "checkpoint")' )

parser.add_argument('--eval_only',
                    action='store_true', default=False,
                    help='evaluation only option')

parser.add_argument('--lr',
                    type=float, default=0.0002,
                    help='learning rate (default: 0.0002)')

parser.add_argument('--max_epoch',
                    type=int, default=200,
                    help='maximum number of epochs (default: 200)')

parser.add_argument('--no-cuda',
                    action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--num_k',
                    type=int, default=4,
                    help='# gradient descent iterations for phi(G(x)) learning (default: 4)')

parser.add_argument('--num_kq',
                    type=int, default=4,
                    help='# gradient descent iterations for q(w) learning (default: 4)')

# parser.add_argument( '--one_step',
#  action='store_true', default=False,
#  help='one step training with gradient reversal layer' )

parser.add_argument('--optimizer',
                    type=str, default='adam',
                    help='optimizer (default: "adam")')

parser.add_argument('--resume_epoch',
                    type=int, default=100,
                    help='epoch to resume (default: 100)')

parser.add_argument('--save_epoch',
                    type=int, default=10,
                    help='when to restore the model (default: 10)')

parser.add_argument('--save_model',
                    action='store_true', default=False,
                    help='save_model or not')

parser.add_argument('--seed',
                    type=int, default=1,
                    help='random seed (default: 1)')

parser.add_argument('--source',
                    type=str, default='svhn',
                    help='source dataset (default: "svhn")')

parser.add_argument('--target',
                    type=str, default='mnist',
                    help='target dataset (default: "mnist")')

parser.add_argument('--use_abs_diff',
                    action='store_true', default=False,
                    help='use absolute difference value as a measurement')

parser.add_argument('--fix_randomness',
                    action='store_true', default=False,
                    help='fix randomness')

parser.add_argument('--vadam',
                    default=False, type=bool,
                    help='Whether to use Variation Adam (default: False)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

if args.fix_randomness:
    import numpy as np

    np.random.seed(10)
    torch.backends.cudnn.deterministic = True


###############################################################################

def main():
    # make a string that describes the current running setup
    num = 0
    vadam_str = "vadam" if args.vadam else "adam"
    run_setup_str = f"{args.source}2{args.target}_k_{args.num_k}_kq_{args.num_kq}_lamb_{args.lamb_marg_loss}_{vadam_str}"

    while os.path.exists(f"record/{run_setup_str}_run_{num}.txt"):
        num += 1
    run_setup_str = f"{run_setup_str}_run_{num}"
    # eg, svhn2mnist_k_4_kq_4_lamb_10.0_run_5

    # set file names for records (storing training stats)
    record_train = f"record/{run_setup_str}.txt"
    record_test = f"record/{run_setup_str}_test.txt"
    if not os.path.exists('record'):
        os.mkdir('record')  # create a folder for records if not exist

    # set the checkpoint dir name (storing model params)
    checkpoint_dir = f'checkpoint/{run_setup_str}'
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')  # create a folder if not exist
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)  # create a folder if not exist

    ####

    # create a solver: load data, create models (or load existing models),
    #   and create optimizers
    solver = Solver(args,
                    source=args.source,
                    target=args.target,
                    nsamps_q=args.nsamps_q,
                    lamb_marg_loss=args.lamb_marg_loss,
                    learning_rate=args.lr,
                    batch_size=args.batch_size,
                    optimizer=args.optimizer,
                    num_k=args.num_k,
                    num_kq=args.num_kq,
                    all_use=args.all_use,
                    checkpoint_dir=checkpoint_dir,
                    save_epoch=args.save_epoch,
                    use_vadam=args.vadam)

    # run it (test or training)
    if args.eval_only:
        solver.test(0)
    else:  # training
        count = 0
        for t in range(args.max_epoch):
            num = solver.train(t, record_file=record_train)
            count += num
            if t % 1 == 0:  # run it on test data every epoch (and save models)
                solver.test(t, record_file=record_test,
                            save_model=args.save_model)
            if count >= 20000 * 10:
                break


###############################################################################

if __name__ == '__main__':
    main()
