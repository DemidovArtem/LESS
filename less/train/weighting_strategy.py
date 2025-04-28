import os

import pandas as pd
from sklearn.preprocessing import StandardScaler


def score_standard_scaling(scores_df):
    scaler = StandardScaler()
    scores_df['score_normalized'] = scaler.fit_transform(scores_df[['score']])
    scores_df['score_normalized'] -= scores_df['score_normalized'].min()
    scores_df['score_normalized'] += 1
    scores_df['score_normalized'] /= scores_df['score_normalized'].sum()


scaling_mapping = {
    'standard': score_standard_scaling
}


def add_weights(train_dataset, data_args, scaling: str = 'standard'):
    scores_path = os.path.join(
        os.path.dirname(data_args.train_files[0]),
        f'scores.csv'
    )
    scores_df = pd.read_csv(scores_path)
    if scaling is not None and scaling != 'none':
        scaling_mapping[scaling](scores_df)
        m = len(train_dataset)
        percentage = 0.05
        n = m / percentage
        scores_df['weight'] = 1 / (n * m * scores_df['score_normalized'])
    else:
        scores_df['weight'] = 1

    weights_map = {
        (row['file_name'], str(row['index'])): row['weight']
        for _, row in scores_df.iterrows()
    }
    weights_list = [
        weights_map[
            (
                f"{train_dataset[index]['dataset']}_influence_score.pt",
                str(train_dataset[index]['id'].split('_')[-1])
            )
        ]
        for index in range(len(train_dataset))
    ]

    train_dataset = train_dataset.add_column("weights", weights_list)

    print('Weights were added to train_dataset!')
    print(train_dataset[0])
    return train_dataset, weights_map
