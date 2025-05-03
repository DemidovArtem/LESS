import os
import pdb
import traceback

import pandas as pd
from sklearn.preprocessing import StandardScaler


def debug():
    print(traceback.format_exc())

    pdb.set_trace()


def score_standard_scaling(scores_df):
    scaler = StandardScaler()
    scores_df['score_normalized'] = scaler.fit_transform(scores_df[['score']])
    scores_df['score_normalized'] -= scores_df['score_normalized'].min()
    scores_df['score_normalized'] += 1
    scores_df['score_normalized'] /= scores_df['score_normalized'].sum()


scaling_mapping = {
    'standard': score_standard_scaling
}


def add_weights(
        train_dataset,
        data_args,
):
    scores_path = os.path.join(
        os.path.dirname(data_args.train_files[0]),
        data_args.scores_file_name
    )
    scores_df = pd.read_csv(scores_path)
    scores_df.columns = [col.strip() for col in scores_df.columns]
    if data_args.scaling is not None and data_args.scaling != 'none':
        scaling_mapping[data_args.scaling](scores_df)
        m = len(train_dataset)
        percentage = 0.05
        n = m / percentage
        scores_df['weight'] = 1 / (n * m * scores_df['score_normalized'])
    else:
        scores_df['weight'] = 1
    try:
        weights_map = {
            (row.get('file_name', row['file name']), str(row['index'])): row['weight']
            for _, row in scores_df.iterrows()
        }
    except:
        print(f'{scores_df.columns=}')
        print(f'{scores_df.head()=}')
        for _, row in scores_df.iterrows():
            print(f'{row=}')
            break
        debug()
    weights_list = []
    for index in range(len(train_dataset)):
        weight_key = (
            f"{train_dataset[index]['dataset']}_influence_score.pt",
            str(train_dataset[index]['id'].split('_')[-1])
        )
        if weight_key not in weights_map:
            weight_key = (
                train_dataset[index]['dataset'],
                str(train_dataset[index]['id'].split('_')[-1])
            )
        weights_list.append(weights_map[weight_key])

    train_dataset = train_dataset.add_column("weights", weights_list)

    print('Weights were added to train_dataset!')
    print(train_dataset[0])
    return train_dataset, weights_map
