from typing import List

import pandas as pd
from dexter.config.constants import Separators

class Evidence:
    """ Data class to hold evidence/context for Question Answering

    Args:
        text : text of evidence passage
        title : title of evidence passage
        idx : index of evidence passage
    """
    def __init__(self, text: str, idx=None, title: str = None):
        self._text = text
        self._idx = idx
        self._title = title

    def text(self):
        return self._text

    def id(self):
        return self._idx

    def title(self):
        return self._title


class TableEvidence(Evidence):

    """ Data class to hold evidence/context for Question Answering in the format of a 2D table

    Args:
        table : 2D table of texts
        title: title of the evidence table
        columns : column names of the table
        idx : index of evidence table
    """
    def __init__(self, table: List[List],columns:List, idx=None, title: str = None):
        self.table = table
        self.columns = columns
        super().__init__(self.convert_to_text(), idx, title)

    def convert_to_text(self):
        return Separators.TABLE_ROW_SEP.join([Separators.TABLE_COL_SEP.join(row) for row in [self.columns]+self.table])

    def to_df(self):
        return pd.DataFrame(self.table, columns=self.columns)

