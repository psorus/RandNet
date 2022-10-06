import numpy as np



def genmat(n,m):
    """
    Generate a random connection matrix of size n x m, where each connection is drawn with replacement
    Each element is 1 as long as not no try draws the connection
    So the probability of a connection is:
    prob_drawn_one_try=1/dim
    prob_drawn_dim_tries=1-(1-1/dim)^dim

    """
    dim=n*m
    prob=1-(1-1/dim)**dim

    mat=np.random.uniform(0,1,(n,m))
    return (mat<prob).astype(float)


if __name__ == "__main__":

    n=7
    m=7
    mat=genmat(n,m)
    from plt import plt
    plt.imshow(mat)
    plt.show()

