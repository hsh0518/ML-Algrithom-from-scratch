import math
import heapq
from collections import Counter, defaultdict

class TfIdfSearcher:
    def __init__(self, docs):
        """
        docs: List[str] 待检索文档集合
        """
        self.docs = [doc.lower().split() for doc in docs]
        self.N = len(self.docs)

        # 计算 DF（Document Frequency）
        self.df = Counter()
        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1

        # 计算 IDF
        self.idf = {term: math.log(self.N / df) for term, df in self.df.items()}

        # 计算每篇文档的 TF-IDF 向量
        self.doc_vecs = []
        for doc in self.docs:
            tf = Counter(doc)
            L = len(doc)
            vec = {term: (tf[term] / L) * self.idf[term] for term in tf}
            # L2 归一化
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {term: v / norm for term, v in vec.items()}
            self.doc_vecs.append(vec)

    def _cosine(self, vec1, vec2):
        # 稀疏向量余弦相似度
        return sum(vec1.get(t, 0) * vec2.get(t, 0) for t in vec1)

    def search(self, query, top_n=3):
        """
        query: 查询语句
        返回 top_n 个相似文档的索引列表
        """
        tokens = query.lower().split()
        tf = Counter(tokens)
        L = len(tokens)
        qvec = {term: (tf[term] / L) * self.idf.get(term, 0) for term in tf}
        # 归一化 query 向量
        norm = math.sqrt(sum(v * v for v in qvec.values()))
        if norm > 0:
            qvec = {t: v / norm for t, v in qvec.items()}

        heap = []
        for i, dvec in enumerate(self.doc_vecs):
            sim = self._cosine(qvec, dvec)
            if len(heap) < top_n:
                heapq.heappush(heap, (sim, i))
            else:
                heapq.heappushpop(heap, (sim, i))

        # 从小顶堆中获取 top_n 排序结果
        return [i for _, i in sorted(heap, reverse=True)]
