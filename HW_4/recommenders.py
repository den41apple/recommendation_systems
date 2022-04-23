import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        self.data = data
        # Подготовим DataFrame с популярными item у каждого пользователя
        self.popularity = self.data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        # Подготовим словари
        self.prepare_dicts()
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit()
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data: pd.DataFrame):        
        # your_code
        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='quantity', # Можно пробовать другие варианты
                                          aggfunc='count', 
                                          fill_value=0)
        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        return user_item_matrix
    
    def prepare_dicts(self):
        """Подготавливает вспомогательные словари"""
        
        userids = self.user_item_matrix.index.values
        itemids = self.user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        self.id_to_itemid = dict(zip(matrix_itemids, itemids))
        self.id_to_userid = dict(zip(matrix_userids, userids))

        self.itemid_to_id = dict(zip(itemids, matrix_itemids))
        self.userid_to_id = dict(zip(userids, matrix_userids))
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    def fit(self, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        self.model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             calculate_training_loss=True,
                                             num_threads=num_threads)
        self.model.fit(csr_matrix(self.user_item_matrix).T.tocsr(),
                       show_progress=True)
        
        return self.model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        
        def get_rec_similar_items(self, x):
            recs = self.model.similar_items(self.itemid_to_id[x], N=2)
            top_rec = recs[1][0]
            return self.id_to_itemid[top_rec]
        
        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        
        popularity = self.popularity[self.popularity['user_id'] == user].copy()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity = popularity[popularity['item_id'] != 999999]
        popularity = popularity.groupby('user_id').head(N)
        popularity.sort_values(by=['user_id','quantity'], ascending=False, inplace=True)
        popularity['similar_recommendation'] = popularity['item_id'].apply(lambda x: get_rec_similar_items(self, x))
        res = popularity['similar_recommendation'].to_list()
        assert len(res) == N, f'Количество рекомендаций != {N}'
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        # your_code
        # Находим похожих пользователей
        similar_users_ = self.model.similar_users(self.userid_to_id[user], N=N+1)[1:]
        res = []
        # Делаем рекомендации для каждого похожего пользователя
        for user in similar_users_:
            # Берем только первую рекомендацию
            rec_for_user = self.get_similar_items_recommendation(user=user[0], N=1)[0]
            res.append(rec_for_user)
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res