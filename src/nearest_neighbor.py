from pathlib import Path

import faiss
import numpy as np

from src.vector_index import build_ivf

indexes_dir = Path('indexes')


class NearestNeighbor:

    def __init__(self, vectors_dir, build_with_gpu: bool = True, dim=384):
        self._dim = dim

        if not Path("populated.index").exists():
            # 1. build index for every npy blocks
            ivf_filepaths = list(indexes_dir.glob('*.index'))
            if len(list(Path(vectors_dir).glob('*.npy'))) != len(list(indexes_dir.glob('*.index'))):
                build_ivf(vectors_dir, dim, indexes_dir, build_with_gpu)

            # 2. merge all ivf indexes into one
            ivfs = []
            for ivf_filepath in ivf_filepaths:
                index = faiss.read_index(str(ivf_filepath), faiss.IO_FLAG_MMAP)
                ivfs.append(index.invlists)
                index.own_invlists = False

            index = faiss.read_index("trained_block.index")
            invlists = faiss.OnDiskInvertedLists(index.nlist, index.code_size, "merged_index.ivfdata")
            ivf_vector = faiss.InvertedListsPtrVector()

            for ivf in ivfs:
                ivf_vector.push_back(ivf)

            ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
            index.ntotal = ntotal  # заменяем листы индекса на объединенные
            index.replace_invlists(invlists)
            faiss.write_index(index, "populated.index")

        # 3. read ready index to memory.
        self._index = faiss.read_index('populated.index', faiss.IO_FLAG_ONDISK_SAME_DIR)

        # 4. clone index to GPU
        if build_with_gpu:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index, co)

        # set how many probes to use. it increases quality
        self._index.nprobe = 16
        faiss.omp_set_num_threads(16)

    def find(self, query_vectors, k=1):
        distances, indexes = self._index.search(query_vectors, k=k)
        distances = np.squeeze(distances, axis=1)
        distances = distances.tolist()
        return distances


if __name__ == '__main__':
    nn = NearestNeighbor(vectors_dir='data', dim=384, build_with_gpu=True)
    import time
    from model import Model

    texts = [
        "In this way, we hypothesize that the topological communities of the network will be instantiations of the speech communities that emerge as a result of a language dynamics in that network.",
        "The first social feature inserted in the model is *trust*: we believe that what links people together is not only their topological proximity (or geographical proximity in a social sense) but their shared linguistic conventions: if they can communicate and agree with each other they will trust each other more than if they cannot communicate, and these levels of trust will guide the next communications.",
        "Trust will be used, then, as a social feature that influences the language interactions.",
        "It will be modeled as the weight of the edge that links both individuals (similarly to the definition of trust in \\[[@pone.0182737.ref007]\\]), in a variation of the NG called *Naming Game with Adaptive Weights* (NG-AW) \\[[@pone.0182737.ref008]\\].",
        "The second inserted social feature is *uncertainty* \\[[@pone.0182737.ref009]\\]: the value of trust is based on communications that already happened, and will guide the preference for the next interactions.",
        "Indeed, in a social scope opinions about others are based on the trust already developed.",
        "However, these opinions can change in time and the (at first) considered trustworthy relations can be deceitful.",
        "For this matter, we introduce a factor for modeling the uncertainty(*ϵ*~*i*~) an individual would have in its constructed trust which naturally decays as the individual gathers knowledge on who to trust based on the outcomes of earlier communications.",
        "In our model, the probability of communication through an edge *p*(*i*, *j*) would then be proportional to *trust*~*ij*~ + *ϵ*~*i*~.",
        "Given that both trust and uncertainty are based on the outcome of a communication, which, in turn, depends on the actual utterance made in the interaction, the last inserted social feature is the *utterance preference* \\[[@pone.0182737.ref010]\\], that rules the choice of a subject for the communication.",
        "We argue that an individual would prefer to communicate with an utterance that has been understood before.",
        "In our model, this corresponds to the agent choosing to communicate preferentially using a word that it believes will lead to a successful interaction.",
        "We call this feature *opinion preference*, since the Naming Game can be interpreted as an opinion dissemination model and this is how we will generally refer to it in the rest of the paper.",
        "In order to investigate the similarity between the speech communities and the existing topological communities, we use the Naming Game-based model we developed as a Community Detection algorithm, where utterances are interpreted as propagating labels and groups sharing the same language conventions are interpreted as communities.",
        "There are many similarities between the proposed model and a label propagation algorithm: both are based on local interactions; start with all nodes in the same state; and the resulting labels (or words) tag the existing communities in the network, each being represented by a unique label (word).",
        "In fact, label propagation algorithms were part of the motivation for this work.",
        "Our first contribution with this paper is showing that, with the insertion of each social parameter, the speech communities will correspond more and more to the topological communities of the analyzed networks.",
        "More specifically, we show how a language agreement dynamics on a social network can reveal its modular structure by reaching a meta-stable state of groups of nodes sharing the same word, representing the existing communities.",
        "We measure how well the speech communities match the real communities by comparing them as we would do for a community detection algorithm.",
        "The meta-stable state happens in non-convergent executions, when words get trapped inside the communities.",
        "Our second contribution is to show that the model with three social features is robust for different networks, when we analyze the influence of the network parameters in the dynamics.",
        "Our third main contribution is applying the model to networks with overlapping communities and realizing that the model is also fit for such cases.",
        "The evolution in time of each incorporated social feature studied here---namely trust, uncertainty and utterance preference---results in the self-organization of edges, nodes and words, respectively.",
    ]
    texts.extend([
        "Our contribution is to show that the model with three social features is robust for different networks, when we analyze the influence of the network parameters in the dynamics.",
        "We measure how well the  match the real communities by comparing them as we would do for a community detection algorithm.",
        "In fact, label propagation algorithms for this work.",
    ])
    model = Model(model_name_or_path='sentence-transformers/all-MiniLM-L12-v2')
    query_vectors = model.predict(texts)

    t1 = time.time()
    distances = nn.find(np.random.rand(110000, 384))
    # distances = nn.find(query_vectors)
    t2 = time.time()
    print(distances)
    print(t2 - t1, 'sec')

    # IVF16384,Flat
    # 10 - 0.6 sec
    # 100 - 4.09 sec
