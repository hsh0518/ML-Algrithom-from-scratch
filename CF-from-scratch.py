#test case:
ratings = [
    # user_id, item_id, rating
    [1, 'A', 5],
    [1, 'B', 3],
    [2, 'A', 4],
    [2, 'C', 2],
    [3, 'A', 2],
    [3, 'B', 5],
    [3, 'C', 4]
]

from collections import defaultdict
import math

class Recommender:
    def __init__(self, ratings):
        self.user_ratings = defaultdict(dict)
        self.item_users = defaultdict(set)
        self.users = set()
        self.items = set()
        for user, item, rating in ratings:
            self.user_ratings[user][item] = rating
            self.item_users[item].add(user)
            self.users.add(user)
            self.items.add(item)

    def cosine_similarity(self, u1, u2):
        r1 = self.user_ratings[u1]
        r2 = self.user_ratings[u2]
        common = set(r1.keys()) & set(r2.keys())
        if not common:
            return 0.0
        num = sum(r1[i] * r2[i] for i in common)
        denom = math.sqrt(sum(r1[i] ** 2 for i in r1)) * math.sqrt(sum(r2[i] ** 2 for i in r2))
        return num / denom if denom else 0.0

    def recommend(self, user_id, k=3):
        scores = defaultdict(float)
        similarity_sums = defaultdict(float)
        for other_user in self.users:
            if other_user == user_id:
                continue
            sim = self.cosine_similarity(user_id, other_user)
            if sim <= 0:
                continue
            for item, rating in self.user_ratings[other_user].items():
                if item not in self.user_ratings[user_id]:
                    scores[item] += sim * rating
                    similarity_sums[item] += sim

        # Normalize by sim: if a lot of low sim user giving high score to a item, this item might have high score, this'll smooth it out
        ranked = [(item, score / similarity_sums[item]) for item, score in scores.items() if similarity_sums[item] > 0]
        ranked.sort(key=lambda x: -x[1])
        return [item for item, _ in ranked[:k]]
