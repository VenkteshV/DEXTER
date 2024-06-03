import re
from typing import Union, List
from math import isclose
class FinQAMatch:
    def _clean_num(self, text:str):
        text= str(text).replace("$","")

        text= str(text).replace("%","")
        text= str(text).replace(",","")

        return text

    def extract_num_from_str(self, answer: str):
        s = self._clean_num(answer)
        r_num = r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)"
        groups = re.findall(r_num, s)
        if len(groups) == 0:
            return None
        num = groups[-1][0]
        if num == '':
            return None
        if '.' in num:
            return float(num)
        return int(num)
    def finqa_equal(self,prediction: Union[bool, float, str],    
                reference: Union[float, str],
                include_percentage: bool = False,
                is_close: float = False) -> bool:
        if prediction is None:
            return False
        elif type(prediction) == bool:
            # bool questions
            if prediction:
                return reference == 'yes'
            else:
                return reference == 'no'

        else:
            # number questions
            gt_result = [reference]
            for item in gt_result:
                try:
                    #print(round(float(prediction), 1),round(float(item), 1))
                    if isclose(abs(round(float(item),3)), abs(round(float(prediction),3)), rel_tol=0.02):
                        return True
                    if round(abs(float(prediction)), 1) == round(abs(float(item)), 1):
                        return True
                except Exception as e:
                    print(e)
                    continue
            return False