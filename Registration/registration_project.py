"""
Project code for image registration topics.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output


def intensity_based_registration_demo(I, Im, use_varying_lr=True, use_t2=True, use_d=False, use_ncc=False, use_rigid=False):

    if use_rigid:
        x = np.array([0., 0., 0.])
        fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)

    else: 
        theta = 0.
        sx = sy = 1.
        shx = shy = 0.
        tx = ty = 0.
        x = np.array([theta, sx, sy, shx, shy, tx, ty])
        if use_ncc:
            fun = lambda x: reg.affine_corr(I, Im, x, return_transform=False)
        else:
            fun = lambda x: reg.affine_mi(I, Im, x, return_transform=False)

        # the similarity function
        # fun = function of x = transformation parameters. 
        # function applies affine transformation to IM with parameters x and 
        # and gives similarity between Im and I
        # we use fun as we want to optimalize this function

    # the learning rate
    num_iter = 150
    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    
    # PLOTS
    fig = plt.figure(figsize=(15,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(131 if use_varying_lr else 121)
    im1 = ax1.imshow(I)              # fixed
    im2 = ax1.imshow(I, alpha=0.7)   # moving
    
    if use_varying_lr:
        ax3 = fig.add_subplot(133, xlim=(0, num_iter), ylim=(0, 0.0011))
        learning_rate_line, = ax3.plot([], [], lw=2, label='Learning rate')
        learning_rate_dots = ax3.scatter([], [], label='Updates')
        ax3.set_ylabel('mu')
        ax3.set_xlabel('Iteration')
        ax3.set_title('Varying Learning Rate')
        ax3.grid()

    ax2 = fig.add_subplot(132 if use_varying_lr else 122)
    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_xlim(0, num_iter)
    if use_ncc:
        ax2.set_ylabel('NCC')
        ax2.set_ylim(0,1)
    else: 
        ax2.set_ylabel('MI')
        ax2.set_ylim(0,2)
    if use_varying_lr and use_t2:
        title = 'Inter-modality Registration with Adaptive Learning Rate' 
    elif not use_varying_lr and use_t2:
        title = 'Inter-modality Registration with Constant Learning Rate'
    elif use_varying_lr and use_d:    
        title = 'Intra-modality Registration with Adaptive Learning Rate'
    elif not use_varying_lr and use_d:
        title = 'Intra-modality Registration with Constant Learning Rate'
    ax2.grid()
    ax2.set_title(title)
        

    # initialize
    mu_in = 1e-3
    mu_memory = {}
    
    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):
        # gradient ascent: g is vector of partial derivatives (gradient) of fun
        g = reg.ngradient(fun, x) 
        x += g*mu_in
        if use_rigid:
            S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True)        
        else:
            if use_ncc:
                S, Im_t, _ = reg.affine_corr(I, Im, x, return_transform=True)
            else:
                S, Im_t, _ = reg.affine_mi(I, Im, x, return_transform=True)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)
        if use_varying_lr:
            if k >= 2:
                sim_valid = similarity[:k+1][~np.isnan(similarity[:k+1].flatten())]
                sim_slice = [[val] for val in sim_valid]
                mu_in, mu_memory = varying_learning_rate(mu_in, sim_slice, mu_memory, k)
        else: 
            mu_in = 1e-3
        print(f"Iteration {k}: Similarity = {similarity[k][0]}, Learning Rate = {mu_in}")
        #debug
        print(f"Iteration {k} | x: {x} | grad: {g} | MI: {S}")

        # how to stop when optimum is reached (to save time)
        length_plateau = 10
        if k >= length_plateau:
            recent_similarities = similarity[k-length_plateau+1 : k+1].flatten()
            for s in recent_similarities:
                changes = np.diff(recent_similarities)
            if np.max(np.abs(changes)) < 0.001 and changes[-1] > 0.9:
                print(f"Early stopping at iteration {k}: No significant improvement.")
                break

        # plot learning rate curve
        if use_varying_lr:
            steps = sorted(mu_memory.keys())
            mu_values = [mu_memory[s] for s in steps]
            learning_rate_line.set_data(steps, mu_values)
            learning_rate_dots.set_offsets(np.c_[steps, mu_values])

        display(fig)

    optimum = max(similarity)
    if use_nnc:
        print(f"The optimum Normalized Cross-Correlation of this registration is: {optimum}")
    else:
        print(f"The optimum Mutual Information of this registration is: {optimum}")

def varying_learning_rate(mu_in, similarity, mu_memory, k):
    mu_0 = 1e-3
    min_mu = 1e-5
    MI_local = 0.8
    plateau_counter = 0
    mu_out = mu_in
    
    if len(mu_memory) == 0:
        mu_memory[0] = mu_0 
        mu_memory[1] = mu_0

    if len(similarity) >= 3: 
        MI_current = similarity[-1][0]
        improvement = MI_current - similarity[-2][0]
        previous = similarity[-2][0]-similarity[-3][0]

        # decrease LR when MI is increasing
        # to avoid overshooting the maximum
        if improvement > 0 and abs(improvement - previous) > 0.02: 
            mu_out = max(mu_in * 0.95, min_mu)
            
        # increase LR when MI drops significantly
        elif improvement < -0.05 and (improvement - previous) < -0.05:
            mu_out = min(mu_in * 1.05, mu_0)

        # how to escape local optima by taking a bigger step in learning rate
        # and how to escape a non-progressing MI
        elif MI_current < MI_local and abs(improvement) < 0.001 and abs(improvement - previous) < 0.001:
            mu_out = min(mu_in * 1.1, mu_0)
        
        # how to escape plateaus
        if abs(improvement) < 0.001:
            plateau_counter += 1
            if plateau_counter > 5:
                mu_out = max(mu_in * 0.9, min_mu)
        else:
            plateau_counter = 0  # reset
            
        mu_memory[k] = mu_out

    # grad_norms.append(np.linalg.norm(g))
    # make update dynamic? = 0.0001 + 0.0003 * (1.0 - MI_current)
                                      
    return mu_out, mu_memory


    

