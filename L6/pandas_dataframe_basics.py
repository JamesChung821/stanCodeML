"""
File: pandas_dataframe_basics.py
Name:
-----------------------------------
This file shows the basic pandas syntax, especially
on DataFrame. DataFrame is a 2D data structure, similar to
what a 2D array looks like (or an Excel document).
We will be practicing creating a pandas DataFrame and
call its attributes and methods
"""

import pandas as pd


def main():
    d = pd.DataFrame({'Name': ['A', 'B', 'c'], 'Age': [20, 10, 30]})
    print('---DataFrame---')
    print(d)
    print('---Get a row---')
    print(d.loc[2])
    print('---Get a column---')
    print(d.Age)
    print('---')
    print(d['Age'])
    print('---shape---')
    print(d.shape)
    print('---specific row > 10---')
    print(d.loc[d.Age > 10])
    print('---for Name column---')
    print(d.loc[d.Age > 10, 'Name'])
    print('---Add new column---')
    d['Qualify'] = pd.Series([True, False, True])   # Series is column
    print(d)
    print('---Add new row---')
    new_row = {'Name': 'D', 'Age': 28, 'Qualify': True}
    d = d.append(new_row, ignore_index=True)    # append is row
    print(d)
    print('---Count non-NAN cells---')
    print(d.count())
    d['Male'] = 0
    # d.loc[d.Age == 10, 'Male'] = 1
    print(d)


if __name__ == '__main__':
    main()
