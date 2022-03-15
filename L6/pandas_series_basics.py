"""
File: pandas_series_basics.py
Name: Cheng-Chu Chung
-----------------------------------
This file shows the basic pandas syntax, especially
on Series. Series is a single column
of data, similar to what a 1D array looks like.
We will be practicing creating a pandas Series and
call its attributes and methods
"""

import pandas as pd


def main():
    s1 = pd.Series([10, 20, 20])    # Create a column
    s2 = s1.append(pd.Series([10, 20, 30]), ignore_index=True)  # Add data to the column
    print('---s1---')
    print(s1)
    print('---s2---')
    print(s2)



if __name__ == '__main__':
    main()
