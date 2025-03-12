from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--epochs",type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of batch-size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print("number of epochs: {}".format(args.epochs))
    print("number of batch_size: {}".format(args.batch_size))