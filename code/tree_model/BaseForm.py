from melody import Tree
from typing import Type


class BaseForm(Tree):
    def __init__(self, name='top'):
        super().__init__()
        self.name = name


class Theme(BaseForm):
    """
    Theme here refers to a complete formal unit containing the set of following elements
        - melodic-motivic content
        - accompanimental texture,
        - supporting harmonic progressions
    Different themes utilize these three elements differently when constructing its children
    """
    def __init__(self):
        super().__init__()
        self.latent_variables = {
            'cadence_type': ''
        }

class Sentence(Theme):
    """
    sentence
    """
    def __init__(self):
        super().__init__()
        self.

class ThemeOperation:
    """
    complete phrase: Initiating,Continuation,Closing
    [2 or 3 split]:
        - cadence conclusiveness (if applies):  (s) -> (w,s) or (w,w,s)
        - 
    """
    is_complete = None

    @staticmethod
    def is_legal(form: Type[BaseForm]):
        pass

    @staticmethod
    def perform(form: Type[BaseForm]):
        pass


class Split(ThemeOperation):
    """
    [2 or 3 split]:
        - cadence conclusiveness (if applies):  (s) -> (w,s) or (w,w,s)
        -
    """

    @staticmethod
    def is_legal(form: Type[BaseForm]):
        pass

    @staticmethod
    def perform(form: Type[BaseForm]):
        pass


class FormAction:
    pass


class FormPolicy:
    pass
