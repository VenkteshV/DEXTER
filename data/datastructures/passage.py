class Passage:
    def __init__(self, text:str, idx,title:str):
        self._text = text
        self._idx = idx
        self._title = title

    def text(self):
        return self._text

    def id(self):
        return self._idx
    
    def title(self):
        return self._title
