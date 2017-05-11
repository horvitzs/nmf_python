import numpy as np
import scipy.sparse as sps
import scipy.optimize as opt
import numpy.linalg as nla
import nonnegfac.matrix_utils as mu
import time
import json
from numpy import random
from nonnegfac.nnls import nnlsm_activeset
from nonnegfac.nnls import nnlsm_blockpivot


class NMF_Base(object):

    """ Base class for NMF algorithms

    Specific algorithms need to be implemented by deriving from this class.
    """
    default_max_iter = 100
    default_max_time = np.inf

    def __init__(self):
        raise NotImplementedError(
            'NMF_Base is a base class that cannot be instantiated')

    def set_default(self, default_max_iter, default_max_time):
        self.default_max_iter = default_max_iter
        self.default_max_time = default_max_time

    def run(self, A, k, init=None, max_iter=None, max_time=None, verbose=0):
        """ Run a NMF algorithm

        Parameters
        ----------
        A : numpy.array or scipy.sparse matrix, shape (m,n)
        k : int - target lower rank

        Optional Parameters
        -------------------
        init : (W_init, H_init) where
                    W_init is numpy.array of shape (m,k) and
                    H_init is numpy.array of shape (n,k).
                    If provided, these values are used as initial values for NMF iterations.
        max_iter : int - maximum number of iterations.
                    If not provided, default maximum for each algorithm is used.
        max_time : int - maximum amount of time in seconds.
                    If not provided, default maximum for each algorithm is used.
        verbose : int - 0 (default) - No debugging information is collected, but
                                    input and output information is printed on screen.
                        -1 - No debugging information is collected, and
                                    nothing is printed on screen.
                        1 (debugging/experimental purpose) - History of computation is
                                        returned. See 'rec' variable.
                        2 (debugging/experimental purpose) - History of computation is
                                        additionally printed on screen.
        Returns
        -------
        (W, H, rec)
        W : Obtained factor matrix, shape (m,k)
        H : Obtained coefficient matrix, shape (n,k)
        rec : dict - (debugging/experimental purpose) Auxiliary information about the execution
        """
        info = {'k': k,

                'alg': str(self.__class__),
                'A_dim_1': A.shape[0],
                'A_dim_2': A.shape[1],
                'A_type': str(A.__class__),
                'max_iter': max_iter if max_iter is not None else self.default_max_iter,
                'verbose': verbose,
                'max_time': max_time if max_time is not None else self.default_max_time}
        if init != None:
            W = init[0].copy()
            H = init[1].copy()
            info['init'] = 'user_provided'
        else:
            W = random.rand(A.shape[0], k)
            H = random.rand(A.shape[1], k)
            info['init'] = 'uniform_random'

        if verbose >= 0:
            print ('[NMF] Running: ')
            print (json.dumps(info, indent=4, sort_keys=True))

        norm_A = mu.norm_fro(A)
        total_time = 0

        if verbose >= 1:
            his = {'iter': [], 'elapsed': [], 'rel_error': []}

        start = time.time()
        # algorithm-specific initilization
        (W, H) = self.initializer(W, H)

        for i in range(1, info['max_iter'] + 1):
            start_iter = time.time()
            # algorithm-specific iteration solver
            (W, H) = self.iter_solver(A, W, H, k, i)
            elapsed = time.time() - start_iter

            if verbose >= 1:
                rel_error = mu.norm_fro_err(A, W, H, norm_A) / norm_A
                his['iter'].append(i)
                his['elapsed'].append(elapsed)
                his['rel_error'].append(rel_error)
                if verbose >= 2:
                    print ('iter:' + str(i) + ', elapsed:' + str(elapsed) + ', rel_error:' + str(rel_error))

            total_time += elapsed
            if total_time > info['max_time']:
                break

        W, H, weights = mu.normalize_column_pair(W, H)

        final = {}
        final['norm_A'] = norm_A
        final['rel_error'] = mu.norm_fro_err(A, W, H, norm_A) / norm_A
        final['iterations'] = i
        final['elapsed'] = time.time() - start

        rec = {'info': info, 'final': final}
        if verbose >= 1:
            rec['his'] = his

        if verbose >= 0:
            print ('[NMF] Completed: ')
            print (json.dumps(final, indent=4, sort_keys=True))
        return (W, H, rec)

    def run_repeat(self, A, k, num_trial, max_iter=None, max_time=None, verbose=0):
        """ Run an NMF algorithm several times with random initial values 
            and return the best result in terms of the Frobenius norm of
            the approximation error matrix

        Parameters
        ----------
        A : numpy.array or scipy.sparse matrix, shape (m,n)
        k : int - target lower rank
        num_trial : int number of trials

        Optional Parameters
        -------------------
        max_iter : int - maximum number of iterations for each trial.
                    If not provided, default maximum for each algorithm is used.
        max_time : int - maximum amount of time in seconds for each trial.
                    If not provided, default maximum for each algorithm is used.
        verbose : int - 0 (default) - No debugging information is collected, but
                                    input and output information is printed on screen.
                        -1 - No debugging information is collected, and
                                    nothing is printed on screen.
                        1 (debugging/experimental purpose) - History of computation is
                                        returned. See 'rec' variable.
                        2 (debugging/experimental purpose) - History of computation is
                                        additionally printed on screen.
        Returns
        -------
        (W, H, rec)
        W : Obtained factor matrix, shape (m,k)
        H : Obtained coefficient matrix, shape (n,k)
        rec : dict - (debugging/experimental purpose) Auxiliary information about the execution
        """
        for t in range(num_trial):
            if verbose >= 0:
                print ('[NMF] Running the {0}/{1}-th trial ...'.format(t + 1, num_trial))
            this = self.run(A, k, verbose=(-1 if verbose is 0 else verbose))
            if t == 0:
                best = this
            else:
                if this[2]['final']['rel_error'] < best[2]['final']['rel_error']:
                    best = this
        if verbose >= 0:
            print ('[NMF] Best result is as follows.')
            print (json.dumps(best[2]['final'], indent=4, sort_keys=True))
        return best

    def iter_solver(self, A, W, H, k, it):
        raise NotImplementedError

    def initializer(self, W, H):
        return (W, H)


class NMF_ANLS_BLOCKPIVOT(NMF_Base):

    """ NMF algorithm: ANLS with block principal pivoting

    J. Kim and H. Park, Fast nonnegative matrix factorization: An active-set-like method and comparisons,
    SIAM Journal on Scientific Computing, 
    vol. 33, no. 6, pp. 3261-3281, 2011.
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        Sol, info = nnlsm_blockpivot(W, A, init=H.T)
        H = Sol.T
        Sol, info = nnlsm_blockpivot(H, A.T, init=W.T)
        W = Sol.T
        return (W, H)


class NMF_ANLS_AS_NUMPY(NMF_Base):

    """ NMF algorithm: ANLS with scipy.optimize.nnls solver
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        if not sps.issparse(A):
            for j in range(0, H.shape[0]):
                res = opt.nnls(W, A[:, j])
                H[j, :] = res[0]
        else:
            for j in range(0, H.shape[0]):
                res = opt.nnls(W, A[:, j].toarray()[:, 0])
                H[j, :] = res[0]

        if not sps.issparse(A):
            for j in range(0, W.shape[0]):
                res = opt.nnls(H, A[j, :])
                W[j, :] = res[0]
        else:
            for j in range(0, W.shape[0]):
                res = opt.nnls(H, A[j, :].toarray()[0,:])
                W[j, :] = res[0]
        return (W, H)


class NMF_ANLS_AS_GROUP(NMF_Base):

    """ NMF algorithm: ANLS with active-set method and column grouping

    H. Kim and H. Park, Nonnegative matrix factorization based on alternating nonnegativity 
    constrained least squares and active set method, SIAM Journal on Matrix Analysis and Applications, 
    vol. 30, no. 2, pp. 713-730, 2008.
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        if it == 1:
            Sol, info = nnlsm_activeset(W, A)
            H = Sol.T
            Sol, info = nnlsm_activeset(H, A.T)
            W = Sol.T
        else:
            Sol, info = nnlsm_activeset(W, A, init=H.T)
            H = Sol.T
            Sol, info = nnlsm_activeset(H, A.T, init=W.T)
            W = Sol.T
        return (W, H)


class NMF_HALS(NMF_Base):

    """ NMF algorithm: Hierarchical alternating least squares

    A. Cichocki and A.-H. Phan, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
    IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences,
    vol. E92-A, no. 3, pp. 708-721, 2009.
    """

    def __init__(self, default_max_iter=100, default_max_time=np.inf):
        self.eps = 1e-16
        self.set_default(default_max_iter, default_max_time)

    def initializer(self, W, H):
        W, H, weights = mu.normalize_column_pair(W, H)
        return W, H

    def iter_solver(self, A, W, H, k, it):
        AtW = A.T.dot(W)
        WtW = W.T.dot(W)
        for kk in range(0, k):
            temp_vec = H[:, kk] + AtW[:, kk] - H.dot(WtW[:, kk])
            H[:, kk] = np.maximum(temp_vec, self.eps)

        AH = A.dot(H)
        HtH = H.T.dot(H)
        for kk in range(0, k):
            temp_vec = W[:, kk] * HtH[kk, kk] + AH[:, kk] - W.dot(HtH[:, kk])
            W[:, kk] = np.maximum(temp_vec, self.eps)
            ss = nla.norm(W[:, kk])
            if ss > 0:
                W[:, kk] = W[:, kk] / ss

        return (W, H)


class NMF_MU(NMF_Base):

    """ NMF algorithm: Multiplicative updating 

    Lee and Seung, Algorithms for non-negative matrix factorization, 
    Advances in Neural Information Processing Systems, 2001, pp. 556-562.
    """

    def __init__(self, default_max_iter=500, default_max_time=np.inf):
        self.eps = 1e-16
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W, H, k, it):
        AtW = A.T.dot(W)
        HWtW = H.dot(W.T.dot(W)) + self.eps
        H = H * AtW
        H = H / HWtW

        AH = A.dot(H)
        WHtH = W.dot(H.T.dot(H)) + self.eps
        W = W * AH
        W = W / WHtH

        return (W, H)

###
class NMF_RANK2(NMF_Base):

    def __init__(self, default_max_iter=500, default_max_time=np.inf):
        self.eps = 1e-16
        self.set_default(default_max_iter, default_max_time)

    def iter_solver(self, A, W ,H ,k ,it):

        # H: n * k

        m = np.shape(A)[0]
        n = np.shape(A)[1]
        tol = 1e-4
        vec_norm = 2.0
        normW = True

        left = H.T.dot(H)
        right = A.dot(H)

        #for  i in range(1,10):
      
        W = self.anls_entry_rank2_precompute(left,right,W);


        norms_W = np.sqrt(np.sum(np.square(W)))
        W = W/norms_W
        left = W.T.dot(W)
        right = A.T.dot(W)

        H = self.anls_entry_rank2_precompute(left, right, H)
        gradH = left.dot(H.T).T - right 
        left = H.T.dot(H)
        right = A.dot(H)

        gradW = W.dot(left) - right

        # if(i == 1):
        #     initgrad = np.sqrt(np.square(nla.norm(gradW[gradW<=0|W>0])) + np.square(nla.norm(gradH<=0|H>0)))
        #     continue;
        # else :
        #     projnorm = np.sqrt(np.square(nla.norm(gradW[gradW<=0|W>0])) + np.square(nla.norm(gradH<=0|H>0)))

        # if(projnorm < tol *initgrad):
        #     break;

        # grad = projnorm / initgrad 

        # print(grad)


        # for  i in range(1,10):
        #     if(nla.matrix_rank(left)<2):
        #         # singular
        #         W = np.zeros(m,2)
        #         H = np.zeros(n,2);
        #         U, S, Vh = nla.svd(A);
        #         print('singular')

        #     W = self.anls_entry_rank2_precompute(left,right,W).;


        if vec_norm !=0:
            if normW :
                norms = np.sum(np.power(W,vec_norm), axis=0)*(1/vec_norm)
                H = H * norms 
                W = W / norms 
            else :
                norms = np.sum(np.power(H,vec_norm),axis = 0)*(1/vec_norm)
                W = W.dot(norms)
             
        # print(np.shape(A))

        # if np.shape(A) !=2:
        #     print ('error');


        return (W,H)


    def anls_entry_rank2_precompute(self, left, right, H):

       #if abs(left(1,1)) < eps & abs(left(1,2)) < eps

       #print (np.shape(left))
       #print (left)

        n = np.shape(right)[0]

        solve_either = np.zeros((n,2)) 
        solve_either[:,0] = right[:,0] * (1./left[0,0])
        solve_either[:,1] = right[:,1] * (1./left[1,1])
        cosine_either = np.zeros((n,2))
        cosine_either[:,0] = np.multiply(solve_either[:,0] , np.sqrt(left[0,0]))
        cosine_either[:,1] = np.multiply(solve_either[:,1] , np.sqrt(left[1,1]))

        choose_first = (cosine_either[:,0] >= cosine_either[:,1])


        solve_either[choose_first,1] = 0
        solve_either[~choose_first,0] = 0

        if ( abs(left[0,0]) <= abs(left[0,1])):

            t = left[1,0]/left[0,0];
            a2 = left[0,0] + t*left[1,0];
            b2 = left[0,1] + t*left[1,1];
            d2 = left[1,1] - t*left[0,1];

            e2 = right[:,0] + t * right[:,1]
            f2 = right[:,1] - t * right[:,0]

        else:

            ct = left[0,0] / left[1,0]
            a2 = left[1,0] + ct * left[0,0]
            b2 = left[1,1] + ct * left[0,1]
            d2 = -left[0,1] + ct * left[1,1]

            e2 = right[:,1] + ct * right[:,0]
            f2 = -right[:,0] + ct * right[:,1]


        H[:,1] = f2 * (1/d2)
        H[:,0] = (e2-b2*H[:,1])*(1/a2)    


        use_either = ~np.all(H>0,1)
        H[use_either,:] = solve_either[use_either,:]


        return H



###
class Hier8_net():

    def __init__(self):
        print('start hier8')

    def hier8_net(self,A,k): 

       # print(np.where(np.sum(A[:,0]))


        trial_allowance = 3 
        unbalance = 0.1 
        vec_norm = 2.0 
        normW = True 
        tol = 1e-4 
        maxiter= 10000 

        m = np.shape(A)[0]
        n = np.shape(A)[1]

        timings = np.zeros((1,k-1))
        #clusters = np.zeros((1,2*(k-1)))
        clusters = []
        #Ws = np.zeros((1,2*(k-1)))
        Ws = []
        W_buffer = [] 
        H_buffer = [] 
        priorities = np.zeros((2*(k-1)))
        is_leaf = -1 * np.ones(2*(k-1))
        tree = np.zeros((2,2*(k-1)))
        splits = -1 * np.ones(k-1)

        term_subset =np.where(np.sum(A, axis=1)!=0) 
        # print(m,n)
        # print(np.shape(term_subset))
        term_subset = np.array(term_subset[0])
        term_subset = term_subset.flatten()
        W = np.random.rand(np.size(term_subset),2)
        H = np.random.rand(n,2)

        print('hier8')

        if(np.size(term_subset) == m):
            W, H = self.nmfsh_comb_rank2(A, W ,H )
            print('done')
        else : 
            W_tmp, H = self.nmfsh_comb_rank2(A[term_subset,:], W, H)
            print('done rank2')
            W = np.zeros((m,2))
            W[term_subset,:] = W_tmp

        result_used = 0


        for i in range(0,k):
            if(i == 0):
                split_node = 0; 
                new_nodes = np.array([0,1])
                min_priority = 1e308
                split_subset = []
            else: 
                leaves = np.where(is_leaf==1)[0]
                #leaves = np.array(leaves)
                print('leaves')
                print(leaves)
                temp_priority = priorities[leaves]
                print('temp_priority', temp_priority)
                print('priorities', priorities)
                #min_priority = np.minimum(temp_priority[temp_priority>0])
                #print('min_priority' + min_priority)
                split_node = leaves[split_node]
                is_leaf[split_node] = 0 
                print('split_node')
                print(split_node)
                W = W_buffer[split_node]
                H = H_buffer[split_node]
                split_subset = clusters[split_node]
                new_nodes = np.array([result_used, result_used+1])
                tree[0,split_node] = new_nodes[0]
                tree[1,split_node] = new_nodes[1]
                print('new nodes', new_nodes)

            result_used = result_used + 2
            max_val, cluster_subset =  H.T.max(0), H.T.argmax(0)  
            #clusters[new_nodes[0]] = np.where(cluster_subset==0)  
            # temp = np.where(cluster_subset == 0 )
            # temp = np.array(temp)
            # temp2 = np.where(cluster_subset == 1)
            # temp2 =np.array(temp2)
            # clusters[new_nodes[0]] = temp
            # clusters.append(temp2)
            clusters.append(np.array(np.where(cluster_subset == 0)))
            clusters.append(np.array(np.where(cluster_subset == 1)))
            #clusters = np.array(clusters)
            Ws.append(W[:,0])
            Ws.append(W[:,1])
            splits[i] = split_node
            is_leaf[new_nodes] = 1 

            #print('length of each clusters', np.shape(clusters[new_nodes[0]]), np.shape(clusters[new_nodes[1]]))

            subset = clusters[new_nodes[0]]
            subset, W_buffer_one, H_buffer_one, priority_one = self.trial_split(trial_allowance, unbalance, min_priority, A, subset, W[:,0])
            print('done trial_split')
            clusters[new_nodes[0]] = subset
            W_buffer.append(W_buffer_one)
            H_buffer.append(H_buffer_one)
            priorities[new_nodes[0]] = priority_one
            print('priority_one', priority_one)

            subset = clusters[new_nodes[1]]
            subset, W_buffer_one, H_buffer_one, priority_one = self.trial_split(trial_allowance, unbalance, min_priority, A, subset, W[:,1])
            clusters[new_nodes[1]] = subset
            W_buffer.append(W_buffer_one)
            H_buffer.append(H_buffer_one)
            priorities[new_nodes[1]] = priority_one

            print('qwe')
              #  else:    
              #      min_priority = min(temp_priority[temp_priority>0])


    def  trial_split(self, trial_allowance, unbalance, min_priority, A, subset, W_parent):

        trial = 0
        subset = np.array(subset)[0]
        subset_backup = subset 
        while(trial < trial_allowance):
            cluster_subset, W_buffer_one, H_buffer_one, priority_one = self.actual_split(A, subset, W_parent)
            if(priority_one < 0 ):
                break;
            unique_cluster_subset  = np.unique(cluster_subset)
            temp = np.where(cluster_subset == unique_cluster_subset[0])
            temp = np.array(temp)
            temp = temp.flatten()
            length_cluster1 = len(temp)

            temp2 = np.where(cluster_subset == unique_cluster_subset[1])
            temp2 = np.array(temp2)
            temp2 = temp2.flatten()
            length_cluster2 = len(temp2)

            if(np.minimum(length_cluster1, length_cluster2) < unbalance * len(cluster_subset)):
                print('dasda')
                min_val = np.minimum(length_cluster1,length_cluster2)
                if (length_cluster1 - length_cluster2 >=0):
                    idx_small = 0
                else:
                    idx_small = 1
                subset_small = np.where(cluster_subset == unique_cluster_subset[idx_small])[0]
                #print(np.shape(subset))
                #print(subset_small)
                subset_small = subset[subset_small]
                cluster_subset_small, W_buffer_one_small, H_buffer_one_small, priority_one_small = self.actual_split(A, subset_small, W_buffer_one[:,idx_small])
                if (priority_one_small < min_priority):
                    trial = trial + 1 
                    if(trial < trial_allowance):
                        subset = np.setdiff1d(subset, subset_small)
                    else: 
                        break;
                else:
                    break; 

                #subset_small = np.array(subset_small)
                #subset_small = subset_small.faltten()
                #print(subset_small)

        if( trial == trial_allowance):

            subset = subset_backup
            W_buffer_one = np.zeros((m,2))
            H_buffer_one = np.zeros(len(subset,2))
            priority_one = -2

        return subset, W_buffer_one, H_buffer_one, priority_one

    def actual_split(self, A, subset, W_parent):

        m = np.shape(A)[0]
        n = np.shape(A)[1]
        #print(np.size(subset))

        if( np.size(subset) <= 3):
            cluster_subset = np.ones((1,len(subset)))
            W_buffer_one = np.zeros((m,2))
            H_buffer_one = np.zeros((len(subset),2))
            priority_one = -1 
        else:
            subset = subset.flatten()
            #print(np.sum(A[:,subset], axis=1))
            #print(np.shape(np.sum(A[:,subset], axis=1)))
            term_subset = np.where(np.sum(A[:,subset], axis=1) !=0)
            term_subset = np.array(term_subset)[0]
            term_subset = term_subset.flatten()
            print('actual_split')
            #print(np.shape(term_subset))
            # print(A[term_subset][:,subset])
            #print(np.shape(A[term_subset][:,subset]))
            A_subset = A[term_subset][:,subset]; 
            W = random.rand(len(term_subset),2)
            H = random.rand(len(subset),2)
            W, H = self.nmfsh_comb_rank2(A_subset, W, H)
            print(np.shape(H))
            max_val, cluster_subset =  H.T.max(0), H.T.argmax(0)  
            W_buffer_one = np.zeros((m,2))
            W_buffer_one[term_subset,:] = W 
            H_buffer_one = H 
            if(len(np.unique(cluster_subset))>1):
                priority_one = self.compute_priority(W_parent, W_buffer_one)
                print('priority_one',priority_one)
            else:
                priority_one = -1 

        return cluster_subset, W_buffer_one, H_buffer_one, priority_one

    def compute_priority(self, W_parent, W_child):


        print('compute_priority')
        n = len(W_parent)
        print(n)
        sorted_parent, idx_parent = np.sort(W_parent)[::-1], np.argsort(W_parent[::-1]) #descending order
        sorted_child1, idx_child1 = -np.sort(-W_child[:,0]), np.argsort(-W_child[:,0])
        sorted_child2, idx_child2 = -np.sort(-W_child[:,1]), np.argsort(-W_child[:,1])

        temp = np.array(np.where(W_parent !=0))
        temp = temp.flatten()
        n_part = len(temp)
    
        if(n_part <= 1):
            priority = -3
        else:
            weight = np.log(np.arange(n,0,-1))
            first_zero = np.where(sorted_parent==0 & 1)[0]
            if(len(first_zero)>0):
                weight[first_zero] = 1 
            weight_part = np.zeros((n,1)).flatten()
            weight_part[0:n_part] = np.log(np.arange(n_part,0,-1))
            sorted1, idx1 = np.sort(idx_child1), np.argsort(idx_child1)
            sorted2, idx2 = np.sort(idx_child2), np.argsort(idx_child2)
          
            max_pos =  np.maximum(idx1, idx2) 
            discount = np.log(n - max_pos[idx_parent]+1)
            discount[discount ==0] = np.log(2)
            weight = weight / discount
            weight_part = weight_part / discount
            print(weight, weight_part)
            priority = self.NDCG_part(idx_parent, idx_child1, weight, weight_part) * self.NDCG_part(idx_parent, idx_child2, weight, weight_part)
        


        return priority        


    def NDCG_part(self, ground, test, weight, weight_part):

        sorted1, seq_idx = np.sort(ground), np.argsort(ground)
        weight_part = weight_part[seq_idx]
        
        n = len(test)
        uncum_score = weight_part[test]
        uncum_score[1:n-1:1] = np.log2(uncum_score[1:n-1:1])
        cum_score = np.cumsum(uncum_score)

        ideal_score = np.sort(weight)[::-1]
        ideal_score[1:n-1:1] = np.log2(ideal_score[1:n-1:1])
        cum_ideal_score = np.cumsum(ideal_score)

        score = cum_score / cum_ideal_score 
        score = score[-1]

        #print(score)

        return score
                


    def nmfsh_comb_rank2(self,A, Winit, Hinit):
            
        m = np.shape(A)[0]
        n = np.shape(A)[1]
        tol = 1e-4
        vec_norm = 2.0
        normW = True

        W = Winit 
        H = Hinit.T

        print('nmfsh_comb_rank2')


        left = H.dot(H.T)
        right = A.dot(H.T)

        #print(np.shape(left), np.shape(right))

        for i in range(0,1000):
            if(nla.matrix_rank(left)<2):
                print('The matrix H is singular')
                W = np.zeors((m,2))
                H = np.zeros((2,n))
                U, S, V = nla.svd(A,1)
                if(np.sum(U)<0):
                    U = - U 
                    V = - V 
                W[:,0] = U 
                H[0,:] = V.T 

      
            W = self.anls_entry_rank2_precompute(left,right,W);

            #print('W shape', np.shape(W))
            norms_W = np.sqrt(np.sum(np.square(W)))
            W = W/norms_W
            left = W.T.dot(W)
            right = A.T.dot(W)
            #print(np.shape(A), np.shape(right))

            H = self.anls_entry_rank2_precompute(left, right, H.T).T
            gradH = left.dot(H) - right.T 
            left = H.dot(H.T)
            right = A.dot(H.T)

            gradW = W.dot(left) - right


        if vec_norm !=0:
            if normW :
                norms = np.sum(np.power(W,vec_norm), axis=0)*(1/vec_norm)
                #norms = np.matrix(norms)
                H[:,0] = H[:,0] * norms[0]
                H[:,1] = H[:,1] * norms[1]
                W[:,0] = W[:,0] / norms[0]
                W[:,1] = W[:,1] / norms[1]
            else :
                norms = np.sum(np.power(H,vec_norm),axis = 0)*(1/vec_norm)
                #norms = np.matrix(norms)
                H[:,0] = H[:,0] * norms[0]
                H[:,1] = H[:,1] * norms[1]
                W[:,0] = W[:,0] / norms[0]
                W[:,1] = W[:,1] / norms[1]

        H = H.T
             


        return (W,H)



    def anls_entry_rank2_precompute(self, left, right, H):


        n = np.shape(right)[0]

        solve_either = np.zeros((n,2)) 
        solve_either[:,0] = right[:,0] * (1./left[0,0])
        solve_either[:,1] = right[:,1] * (1./left[1,1])
        cosine_either = np.zeros((n,2))
        cosine_either[:,0] = np.multiply(solve_either[:,0] , np.sqrt(left[0,0]))
        cosine_either[:,1] = np.multiply(solve_either[:,1] , np.sqrt(left[1,1]))

        choose_first = (cosine_either[:,0] >= cosine_either[:,1])


        solve_either[choose_first,1] = 0
        solve_either[~choose_first,0] = 0

        if ( abs(left[0,0]) <= abs(left[0,1])):

            t = left[1,0]/left[0,0];
            a2 = left[0,0] + t*left[1,0];
            b2 = left[0,1] + t*left[1,1];
            d2 = left[1,1] - t*left[0,1];

            e2 = right[:,0] + t * right[:,1]
            f2 = right[:,1] - t * right[:,0]

        else:

            ct = left[0,0] / left[1,0]
            a2 = left[1,0] + ct * left[0,0]
            b2 = left[1,1] + ct * left[0,1]
            d2 = -left[0,1] + ct * left[1,1]

            e2 = right[:,1] + ct * right[:,0]
            f2 = -right[:,0] + ct * right[:,1]


        H[:,1] = f2 * (1/d2)
        H[:,0] = (e2-b2*H[:,1])*(1/a2)    


        use_either = ~np.all(H>0,1)
        H[use_either,:] = solve_either[use_either,:]


        return H







class NMF(NMF_ANLS_BLOCKPIVOT):

    """ Default NMF algorithm: NMF_ANLS_BLOCKPIVOT
    """

    def __init__(self, default_max_iter=50, default_max_time=np.inf):
        self.set_default(default_max_iter, default_max_time)


def _mmio_example(m=100, n=100, k=10):
    print ('\nTesting mmio read and write ...\n')
    import scipy.io.mmio as mmio

    W_org = random.rand(m, k)
    H_org = random.rand(n, k)
    X = W_org.dot(H_org.T)
    X[random.rand(n, k) < 0.5] = 0
    X_sparse = sps.csr_matrix(X)

    filename = '_temp_mmio.mtx'
    mmio.mmwrite(filename, X_sparse)
    A = mmio.mmread(filename)

    alg = NMF_ANLS_BLOCKPIVOT()
    rslt = alg.run(X_sparse, k, max_iter=50)


def _compare_nmf(m=300, n=300, k=10):
    from pylab import plot, show, legend, xlabel, ylabel

    W_org = random.rand(m, k)
    H_org = random.rand(n, k)
    A = W_org.dot(H_org.T)

    print ('\nComparing NMF algorithms ...\n')

    names = [NMF_MU, NMF_HALS, NMF_ANLS_BLOCKPIVOT,
             NMF_ANLS_AS_NUMPY, NMF_ANLS_AS_GROUP]
    iters = [2000, 1000, 100, 100, 100]
    labels = ['mu', 'hals', 'anls_bp', 'anls_as_numpy', 'anls_as_group']
    styles = ['-x', '-o', '-+', '-s', '-D']

    results = []
    init_val = (random.rand(m, k), random.rand(n, k))

    for i in range(len(names)):
        alg = names[i]()
        results.append(
            alg.run(A, k, init=init_val, max_iter=iters[i], verbose=1))

    for i in range(len(names)):
        his = results[i][2]['his']
        plot(np.cumsum(his['elapsed']), his['rel_error'],
             styles[i], label=labels[i])

    xlabel('time (sec)')
    ylabel('relative error')
    legend()
    show()


def _test_nmf(m=300, n=300, k=10):
    W_org = random.rand(m, k)
    H_org = random.rand(n, k)
    A = W_org.dot(H_org.T)

    alg_names = [NMF_ANLS_BLOCKPIVOT, NMF_ANLS_AS_GROUP,
                 NMF_ANLS_AS_NUMPY, NMF_HALS, NMF_MU]
    iters = [50, 50, 50, 500, 1000]

    print ('\nTesting with a dense matrix...\n')
    for alg_name, i in zip(alg_names, iters):
        alg = alg_name()
        rslt = alg.run(A, k, max_iter=i)

    print ('\nTesting with a sparse matrix...\n')
    A_sparse = sps.csr_matrix(A)
    for alg_name, i in zip(alg_names, iters):
        alg = alg_name()
        rslt = alg.run(A_sparse, k, max_iter=i)


if __name__ == '__main__':
    _test_nmf()
    _mmio_example()

    # To see an example of comparisons of NMF algorithms, execute
    # _compare_nmf() with X-window enabled.
    # _compare_nmf()
