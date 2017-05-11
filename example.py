from numpy import random
from nonnegfac.nmf import NMF
from nonnegfac.nmf import NMF_ANLS_BLOCKPIVOT
from nonnegfac.nmf import NMF_ANLS_AS_NUMPY
from nonnegfac.nmf import NMF_ANLS_AS_GROUP
from nonnegfac.nmf import NMF_HALS
from nonnegfac.nmf import NMF_MU
from nonnegfac.nmf import NMF_RANK2
from nonnegfac.nmf import Hier8_net
import time
import numpy as np 
from scipy import sparse

if __name__ == '__main__':
    W_org = random.rand(300, 10)
    H_org = random.rand(300, 10)
    A = W_org.dot(H_org.T)
    # print ('\nTesting NMF().run() ...\n')
    # W, H, info = NMF().run(A, 10)
    # print ('\nTesting NMF().run_repeat() ...\n')
    # W, H, info = NMF().run_repeat(A, 10, 10)

    # print ('\nTesting NMF_ANLS_BLOCKPIVOT ...\n')
    # W, H, info = NMF_ANLS_BLOCKPIVOT().run(A, 10, max_iter=50)
    # print ('\nTesting NMF_ANLS_AS_NUMPY ...\n')
    # W, H, info = NMF_ANLS_AS_NUMPY().run(A, 10, max_iter=50)
    # print ('\nTesting NMF_ANLS_AS_GROUP ...\n')
    # W, H, info = NMF_ANLS_AS_GROUP().run(A, 10, max_iter=50)
    start_time = time.time()

    print ('\nTesting NMF_HALS ...\n')
    W, H, info = NMF_HALS().run(A, 2, max_iter=500)

    elaped_time = time.time() - start_time
    print (elaped_time)

    start_time2 = time.time()

    print ('\nTesting NMF_RANK2....\n')
    # W, H, info = NMF_RANK2().run(A, 2, max_iter=1000)

    # elaped_time2 = time.time() - start_time2
    # print (elaped_time2)

    voca_file = open('nonnegfac/tmp1/vocabulary.txt', 'r', encoding = 'UTF8')

    # for line in voca_file:
    #     print(line)

    #tdm_file = open('nonnegfac/tmp1/tmp.mtx', 'r', encoding = 'UTF8')
    tdm = []
    with open('nonnegfac/tmp1/tmp.mtx', 'r', encoding = 'UTF8') as f:
        lines = f.readlines()
        for line in lines:
            v = line[:-1].split()
            item = np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.double)
            tdm =np.append(tdm, item, axis=0)
    tdm = np.array(tdm, dtype=np.double).reshape(len(lines), 3)

    # print(np.shape(tdm[:,0]))
    # print(np.shape(tdm[:,0].max(0)))
    # print(tdm[:,0].max(0))
    # print(tdm[:,1].max(0))

    A = sparse.csr_matrix((tdm[:,2], (tdm[:,0], tdm[:,1])), shape=(int(tdm[:,0].max(0)+1), int(tdm[:,1].max(0)+1)))
    #A = sparse.csr_matrix((tdm[:,0], (tdm[:,1], tdm[:,2])), shape= (5000,500))
    #W, H, info = NMF_RANK2().run(A,2,max_iter=500)
    W, H, info = Hier8_net().hier8_net(A, 4)





    # print ('\nTesting NMF_Hier8....\n')
    # W, H, info = Hier8_net().hier8_net(A, 4)



    # print ('\nTesting NMF_MU ...\n')
    # W, H, info = NMF_MU().run(A, 10, max_iter=500)
