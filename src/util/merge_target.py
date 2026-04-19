

def merge_target(df,X_unlabeled):
    # merge the "target" column into the df
    df_merged = df.merge(X_unlabeled[["ID","target"]], on='ID', how='left',suffixes=('_x', '_y'))
    df_merged['target'] = df_merged['target_y'].fillna(df_merged['target_x'])

    df_final = df_merged.drop(['target_x', 'target_y'], axis=1)

    return df_final
