import pandas as pd
import pandas.testing as pdt

from data_cleaning import filter_smpep2_out


def test_filter_smpep2_out_basic():
    df = pd.DataFrame({ 'SMPEP2': [-10, 0, 1, 550, 551, 1000, None] })

    mask = filter_smpep2_out(df)

    # Expect only values in the open-closed interval (0, 550] to be True
    expected = pd.Series([False, False, True, True, False, False, False])

    pdt.assert_series_equal(mask, expected, check_names=False)
