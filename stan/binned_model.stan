functions{

    matrix get_cov(real h, real w) {
        matrix[2, 2] C;
        C[1, 1] = square(h);
        C[2, 1] = 0.;
        C[1, 2] = 0.;
        C[2, 2] = square(w);
        return C;
    }

    matrix get_R(int i, int i0, int N, vector x, vector y) {
        // Compute the node rotation matrix at the i-th point
        matrix[2, 2] R;
        vector[2] dxy;
        real theta;
        int j;

        j = i + i0;
        if (i == 1) {
            dxy[1] = (x[j+1] - x[j]);
            dxy[2] = (y[j+1] - y[j]);
        } else if (i == N) {
            dxy[1] = (x[j] - x[j-1]);
            dxy[2] = (y[j] - y[j-1]);
        } else {
            dxy[1] = (x[j+1] - x[j-1]);
            dxy[2] = (y[j+1] - y[j-1]);
        }
        dxy /= sqrt(sum(square(dxy)));
        theta = atan2(dxy[2], dxy[1]);

        R[1, 1] = cos(theta);
        R[2, 2] = R[1, 1];
        R[1, 2] = -sin(theta);
        R[2, 1] = sin(theta);

        return R;
    }

}

data {
    // number of pixels in the density map
    int n_pix;
    // number counts of objects in pixels
    int hh[n_pix];

    // selection function for each pixel
    real log_S[n_pix];

    // the x locations of pixels
    vector[n_pix] x;
    // the y locations of pixels
    vector[n_pix] y;

    // ----------------------------------------------------------
    // Stream nodes:

    // number of nodes
    int t_n_nodes;
    int l_n_nodes;

    // nodes locations along rigid polynomial
    vector[t_n_nodes + l_n_nodes] phi1_nodes;
    vector[t_n_nodes + l_n_nodes] phi2_nodes_init;

    // width of nodes: along rigid polynomial, h
    vector[t_n_nodes + l_n_nodes] h_nodes;

    // ----------------------------------------------------------
    // Background nodes:

    // number of nodes
    int n_bg_nodes;

    // width and height standard deviations
    real bg_h;
    real bg_w;

    // nodes locations along rigid polynomial
    vector[n_bg_nodes] bg_phi1_nodes;
    vector[n_bg_nodes] bg_phi2_nodes;
}

transformed data {
    int n_nodes = t_n_nodes + l_n_nodes;

    matrix[2,2] C_bg;

    vector[2] bg_node_xy;
    vector[2] bg_xy;
    vector[n_pix] bg_ln_norm[n_bg_nodes];

    // HACK:
    // vector<lower=-0.3, upper=0.3>[n_nodes] d_phi2_nodes;
    // for (i in 1:n_nodes) {
    //     d_phi2_nodes[i] = 0.;
    // }

    C_bg = get_cov(bg_h, bg_w);
    for (i in 1:n_pix) {
        bg_xy[1] = x[i];
        bg_xy[2] = y[i];
        for (j in 1:n_bg_nodes) {
            bg_node_xy[1] = bg_phi1_nodes[j];
            bg_node_xy[2] = bg_phi2_nodes[j];

            bg_ln_norm[j][i] = multi_normal_lpdf(bg_xy | bg_node_xy, C_bg);
        }
    }
}

parameters {
    // Stream model:
    vector<lower=-0.3, upper=0.3>[n_nodes] d_phi2_nodes;
    vector<lower=-2.5, upper=0.>[n_nodes] log_w_nodes;
    vector<lower=-8, upper=8>[n_nodes] log_a_nodes;

    // Background:
    real<lower=1e-1, upper=1e3> bg_val;
    vector<lower=-8, upper=8>[n_bg_nodes] bg_log_a_nodes;
}

transformed parameters {
    vector[n_pix] log_bg_int;
    vector[n_pix] log_gd1_int;
    vector[n_pix] xmod;

    vector[n_nodes] tmp;
    vector[n_bg_nodes] tmp_bg;
    vector[2] xy;
    vector[2] node_xy;

    vector[n_nodes] phi2_nodes;

    matrix[2, 2] R[n_nodes];

    real ln_bg_val;

    // Un-log some things
    vector[n_nodes] w_nodes;
    matrix[2, 2] C[n_nodes];

    // Background
    ln_bg_val = log(bg_val);

    w_nodes = exp(log_w_nodes);
    phi2_nodes = phi2_nodes_init + d_phi2_nodes;

    // Re-compute node rotation matrices
    for (i in 1:t_n_nodes) {
        R[i] = get_R(i, 0, t_n_nodes, phi1_nodes, phi2_nodes);
    }
    for (i in 1:l_n_nodes) {
        R[t_n_nodes + i] = get_R(i, t_n_nodes, l_n_nodes,
                                 phi1_nodes, phi2_nodes);
    }

    for (i in 1:n_pix) {
        xy[1] = x[i];
        xy[2] = y[i];
        for (j in 1:n_nodes) {
            node_xy[1] = phi1_nodes[j];
            node_xy[2] = phi2_nodes[j];

            // Get the local covariance matrix and rotate to tangent space
            C[j] = get_cov(h_nodes[j], w_nodes[j]);
            C[j] = R[j] * C[j] * R[j]'; // R C R_T
            tmp[j] = log_a_nodes[j] + multi_normal_lpdf(xy | node_xy, C[j]);
        }
        log_gd1_int[i] = log_sum_exp(tmp);

        for (j in 1:n_bg_nodes) {
            tmp_bg[j] = bg_log_a_nodes[j] + bg_ln_norm[j][i];
        }
        log_bg_int[i] = log_sum_exp(ln_bg_val, log_sum_exp(tmp_bg));
        xmod[i] = log_sum_exp(log_gd1_int[i], log_bg_int[i]) + log_S[i];
      }
}
model {
    // Priors
    for (n in 1:n_nodes) {
        log_w_nodes[n] ~ normal(-1, 0.2)T[-3, 0.];
        d_phi2_nodes[n] ~ normal(0, 0.1)T[-0.5, 0.5];
    }

    // Background priors
    bg_val ~ uniform(1e-1, 1e3);

    // like a regularization term to force amplitudes to 0
    // target += -log_a_nodes;
    target += -sum(bg_log_a_nodes);
    // target += -sum(log_w_nodes);

    //Likelihood
    hh ~ poisson_log(xmod);
}
