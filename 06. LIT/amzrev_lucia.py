# Lint as: python3
"""Text classification datasets, including single- and two-sentence tasks."""
from typing import List

from absl import logging
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types


import pandas as pd
import numpy as np


def load_amazonData(*args, **kw):
  """Load from local? """
  # Materialize to NumPy arrays.

  path = '/Users/Documents/Projects/elena/nlp_transformers/data/test/dataset_en_train'
  df = pd.read_json(path, lines = True).rename(columns={'stars':'star_rating'})
  df = df.sample(frac=1).reset_index(drop=True)
  df['text_title'] = df['review_title'] + " " + df['review_body']
  ret = df.to_numpy() ## necesita arrays numpy 

  return ret



class AMAZONes_5cat(lit_dataset.Dataset):
  
  LABELS = ['0','1','2','3','4']

  def __init__(self, path):
    df = pd.read_json(path, lines = True).rename(columns={'stars':'star_rating'})
    df.loc[:,'star_rating']=df.star_rating.apply(lambda x: str(x-1))
    df = df.sample(100)

    self._examples= [{
          'sentence': row['review_body'],
          'review_title': row['review_title'],
          'label': row['star_rating'],

    }for _, row in df.iterrows()]

  def spec(self):
    return {
        'sentence': lit_types.TextSegment(),
        'review_title': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }



class AMAZONes_31cat(lit_dataset.Dataset):
  
  LABELS = ['0','1','2','3','4','5','6','7','8','9','10','11', '12', '13', '14', '15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']

  def __init__(self, path):
    df = pd.read_csv(path)
    df = df.sample(100)

    self._examples= [{
          'sentence': row['review_body'],
          'review_title': row['review_title'],
          'product_name': row['product_category'], 
          'label': row['product_category_cod'],

    }for _, row in df.iterrows()]

  def spec(self):
    return {
        'sentence': lit_types.TextSegment(),
        'review_title': lit_types.TextSegment(),
        'product_name': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }



class CHATBOTes_30cat(lit_dataset.Dataset):
  
  LABELS = ['0','1','2','3','4','5','6','7','8','9','10','11', '12', '13', '14', '15','16','17','18','19','20','21','22','23','24','25','26','27','28','29']

  def __init__(self, path):
    df = pd.read_csv(path)

    self._examples= [{
          'sentence': row['Query'],
          'category_name': row['kb_label'],
          'label': row['category_cod'],

    }for _, row in df.iterrows()]

  def spec(self):
    return {
        'sentence': lit_types.TextSegment(),
        'category_name': lit_types.TextSegment(),
        'label': lit_types.CategoryLabel(vocab=self.LABELS)
    }





