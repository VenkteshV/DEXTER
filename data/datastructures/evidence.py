from typing import List

from constants import Separators


class Evidence:
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
    def __init__(self, table: List, idx=None, title: str = None):
        self.table = table
        super().__init__(self.convert_to_text(), idx, title)

    def convert_to_text(self):
        return Separators.TABLE_ROW_SEP.join([Separators.TABLE_COL_SEP.join(row) for row in self.table])
