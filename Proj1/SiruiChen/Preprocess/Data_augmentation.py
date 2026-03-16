import random
from textaugment import EDA

class DataAugmentation:
    def __init__(self):
        self.augmenter = EDA()
    
    def augment_text(self, text, label, num_aug=2):
        augmented = []
        for _ in range(num_aug):
            # 同义词替换
            aug_text = self.augmenter.synonym_replacement(text)
            augmented.append((aug_text, label))
            
            # 随机插入
            aug_text = self.augmenter.random_insertion(text)
            augmented.append((aug_text, label))
        
        return augmented