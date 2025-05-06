import argparse
import json
import os

import torch


def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_file_names', type=str,
                           nargs='+', help='The path to the score file')
    argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--target_task_names', type=str,
                           nargs='+', help='The name of the target task')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=None,
                           help='The percentage of the data to be selected')
    argparser.add_argument('--threshold', type=float, default=None,
                           help='The threshold for total score of the selected samples. '
                                'Used for iterative building of train set. Should be specified with percentage - '
                                'the latter is used as overall boundary for iteration.')
    argparser.add_argument('--num_iterations', type=int, default=None,
                           help='The number of iterations to collect top-k in case we do it iteratively.')
    argparser.add_argument('--iteration', type=int, default=None,
                           help='Index of current iteration.')

    args = argparser.parse_args()

    return args


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count


if __name__ == "__main__":
    args = parse_args()
    try:
        assert len(args.train_file_names) == len(args.train_files)
    except:
        print(f'{args.train_file_names=}')
        print(f'{args.train_files=}')
        raise
    assert args.percentage is not None or args.max_samples is not None
    assert args.threshold is not None or args.num_iterations is not None
    assert args.iteration is not None or args.num_iterations is None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train_files = len(args.train_file_names)

    for target_task in args.target_task_names:
        output_path = os.path.join(args.output_path, target_task)

        score_paths = [os.path.join(
            output_path, f"{task_name}_influence_score.pt") for task_name in args.train_file_names]
        num_samples = []
        for score_path in score_paths:
            num_samples.append(
                len(torch.load(score_path, map_location=device))
            )
        cumsum_num_samples = torch.cumsum(torch.tensor(num_samples), dim=0)

        total_samples = sum(num_samples)

        prev_top_values = {}
        if args.percentage is not None:
            args.max_samples = int(args.percentage * total_samples)
            if args.threshold is None and args.num_iterations is None:
                data_amount_name = f"p{args.percentage}"
            else:
                for _, _, files in os.walk(output_path):
                    for file in files:
                        if file.startswith('top_') and file.endswith('.jsonl'):
                            file_index = file.split('_')[-1].split('.')[0]  # last value after _ and before extension
                            file_index = int(file_index)
                            # skip files from past runs (same iteration index or bigger)
                            if file_index >= args.iteration:
                                continue
                            with open(os.path.join(output_path, file), 'r') as past_file:
                                prev_top_values[file_index] = [json.loads(line)['id'] for line in past_file]

                # max_samples is reduced by the number of already selected items
                prev_values_count = sum([len(value) for value in prev_top_values.values()])
                if args.threshold:
                    args.max_samples -= prev_values_count
                    data_amount_name = f't{args.threshold}_{args.iteration}'
                else:
                    data_amount_name = f'p{args.percentage}_i{args.num_iterations}_{args.iteration}'
                    args.max_samples = min(args.max_samples - prev_values_count,
                                           args.max_samples // args.num_iterations)
        else:
            data_amount_name = f"num{args.max_samples}"

        all_scores = []
        for score_path, train_file in zip(score_paths, args.train_files):
            score = torch.load(score_path, map_location=device)
            all_scores.append(score)
        all_scores = torch.cat(all_scores, dim=0)

        # sort the scores and output the corresponding data index
        file_specific_index = torch.cat(
            [torch.arange(line_num) for line_num in num_samples]).to(device)
        data_from = torch.cat([torch.ones(line_num, dtype=torch.long)
                               * i for i, line_num in enumerate(num_samples)]).to(device)
        sorted_scores, sorted_index = torch.sort(
            all_scores, dim=0, descending=True
        )
        sorted_file_name = 'sorted.csv'
        if args.threshold:
            sorted_file_name = f'sorted_t{args.threshold}_{args.iteration}.csv'
        elif args.num_iterations:
            sorted_file_name = f'sorted_p{args.percentage}_i{args.num_iterations}_{args.iteration}.csv'
        sorted_score_file = os.path.join(output_path, sorted_file_name)

        data_from = data_from[sorted_index]
        sorted_index = file_specific_index[sorted_index]

        if not os.path.exists(sorted_score_file):
            with open(sorted_score_file, 'w', encoding='utf-8') as file:
                file.write("file name, index, score\n")
                for score, index, name in zip(sorted_scores, sorted_index, data_from):
                    file.write(
                        f"{args.train_file_names[name.item()]}, {index.item()}, {round(score.item(), 6)}\n"
                    )

        all_lines = []
        for i, train_file in enumerate(args.train_files):
            with open(train_file, 'r', encoding='utf-8', errors='ignore') as file:
                all_lines.append(file.readlines()[:num_samples[i]])

        final_index_list = sorted_index.tolist()
        final_data_from = data_from.tolist()
        final_scores = sorted_scores.tolist()

        total_score = 0
        values_added = 0
        result_file_path = os.path.join(output_path, f"top_{data_amount_name}.jsonl")
        with (open(
                file=result_file_path,
                mode='w',
                encoding='utf-8',
                errors='ignore'
        ) as file):
            for index, data_from, score in zip(final_index_list, final_data_from, final_scores):
                try:
                    for prev_values in prev_top_values.values():
                        if json.loads(all_lines[data_from][index])['id'] in prev_values:
                            continue
                    file.write(all_lines[data_from][index])
                    total_score += score
                    values_added += 1
                    if (
                            (args.threshold is not None and total_score >= args.threshold)
                            or values_added == args.max_samples
                    ):
                        break
                except:
                    import traceback

                    print(traceback.format_exc())
                    import pdb

                    pdb.set_trace()
        print(f'Selected data for training were saved to file {result_file_path}')
