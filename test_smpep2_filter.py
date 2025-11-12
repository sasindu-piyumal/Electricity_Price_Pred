import pandas as pd
import numpy as np
import pandas.testing as pdt
from cleaning import filter_smpep2


def test_filter_smpep2():
    # Create a DataFrame with SMPEP2 values including edge cases
    data = {
        'SMPEP2': [-5, 0, 10, 550, 600]
    }
    df = pd.DataFrame(data)

    # Apply the filter
    filtered_df = filter_smpep2(df)

    # Expected result: Only rows with SMPEP2 > 0 and <= 550
    expected_df = pd.DataFrame({'SMPEP2': [10, 550]})

    # Reset index for comparison
    pdt.assert_frame_equal(filtered_df.reset_index(drop=True), expected_df)

if __name__ == '__main__':
    test_filter_smpep2()
    print('Test passed!')
