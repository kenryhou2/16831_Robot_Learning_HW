import argparse
import glob
import os
import csv
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def get_section_results_from_file(file):
    X = []
    Y = []

    current_x = None
    current_y = None

    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                current_x = v.simple_value
            elif v.tag == 'Train_AverageReturn':
                current_y = v.simple_value

        # If both were seen for this event, save them as a pair
        if current_x is not None and current_y is not None:
            X.append(current_x)
            Y.append(current_y)
            current_x = None
            current_y = None

    return X, Y


def get_section_results_from_logdir(logdir):
    eventfiles = sorted(glob.glob(os.path.join(logdir, 'events*')))
    if not eventfiles:
        raise FileNotFoundError(f'No TensorBoard event files found in {logdir}')

    print("Found event files:")
    for f in eventfiles:
        print(f"  {f}")

    all_pairs = []

    for eventfile in eventfiles:
        X, Y = get_section_results_from_file(eventfile)
        all_pairs.extend(zip(X, Y))

    # Sort by timestep
    all_pairs.sort(key=lambda p: p[0])

    # Remove duplicate timesteps, keeping the last value seen
    dedup = {}
    for x, y in all_pairs:
        dedup[x] = y

    X = sorted(dedup.keys())
    Y = [dedup[x] for x in X]

    return X, Y


def save_results_to_csv(csv_path, X, Y):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Train_EnvstepsSoFar', 'Train_AverageReturn'])
        for i, (x, y) in enumerate(zip(X, Y)):
            writer.writerow([i, int(x), y])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='path to directory containing tensorboard results'
    )
    parser.add_argument(
        '--csv_out',
        type=str,
        default=None,
        help='optional output csv path; default is <logdir>/results.csv'
    )
    args = parser.parse_args()

    X, Y = get_section_results_from_logdir(args.logdir)

    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

    csv_path = args.csv_out if args.csv_out is not None else os.path.join(args.logdir, 'results.csv')
    save_results_to_csv(csv_path, X, Y)

    print(f'\nSaved CSV to: {csv_path}')