
# Note code
## Pandas
1. pd.reset_index(drop=True)
   1. Generate a new DataFrame or Series with new index, 
   2. drop=True to without inserting col index to new col in new DataFrame.
2. Drop row have nan number in any column of DataFrame
   ```python
        idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
        idx = np.where(idx)[0]
        annot = annot.drop(idx)
   ```
3. DataFrame get one column by df["col label"] -> Series(col index and col label)
4. df.get_dummies() : Convert categorical variable into dummy/indicator variables.(label one hot)
5. df.drop(): delete one column in df.
6. df.join(): append new data column to df.
6. df.append(): append new data rows to df.
7. Get one or many column in df: df[name_col], df[[name_col1, name_col2, ...]]
8. Convert column data frame to numpy array: df[name_col].values
## Numpy
1. Get value in numpy array 3-dimension: xys[:, :, :2]