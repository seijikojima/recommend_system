import numpy as np

n_users = 5
n_items = 4
# initialize feedback matrix
R = np.array([[5, 2, 0, 0], [1, 1, 0, 0], [0, 2, 4, 3], [5, 1, 2, 0], [5, 3, 1, 0]])
ratings = [] # 評価値のある13のuser_id, item_id, ratingを保持
for user in range(n_users):
    for item in range(n_items):
        if R[user,item] != 0:
            ratings.append([user, item, R[user,item]])

print(ratings)

# R
# array([[5, 2, 0, 0],
#       [1, 1, 0, 0],
#       [0, 2, 4, 3],
#       [5, 1, 2, 0],
#       [5, 3, 1, 0]])
alpha = 0.0002
lambda_ = 0.02
k = 3
p_users = np.random.rand(n_users, k)
q_items = np.random.rand(n_items, k)
for iter in range(10000):
    for user, item, rating in ratings:
        _pred = p_users[user] @ q_items[item] # init : randomで作成したp_users, q_itemsのあるuser_id, item_idの値を予測値
        err = rating - _pred # 実際のratingと予測値の誤差
        p_users[user,:] += alpha * (2.0 * err * q_items[item,:] - lambda_ * p_users[user,:]) # 勾配法でp_users, q_items更新
        q_items[item,:] += alpha * (2.0 * err * p_users[user,:] - lambda_ * q_items[item,:])

print(p_users @ q_items.T)
#  [[ 4.99679368  1.97846771  1.80619416  1.90240633]
#  [ 1.00532748  0.97694863 -0.09778591  0.30744267]
#  [ 8.94208881  1.97955001  3.93211154  3.00003764]
#  [ 4.93061087  1.06020728  2.09390243  1.54747828]
#  [ 4.98767916  2.97632775  1.0093586   1.77344618]]

E = p_users @ q_items.T
print("user_id 2, item_id 0の評価値", E[2,0])