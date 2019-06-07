functions{

	matrix get_cov(real h, real w) {
	    matrix[2, 2] C;
		C[1, 1] = square(h);
		C[2, 1] = 0.;
		C[1, 2] = 0.;
		C[2, 2] = square(w);
		return C;
	}

	matrix get_R(int i, int N, vector x, vector y) {
        matrix[2,2] R;
        vector[2] dxy;
        real theta;

        if (i == 1) {
            dxy[1] = (x[i+1] - x[i]);
            dxy[2] = (y[i+1] - y[i]);
        } else if (i == N) {
            dxy[1] = (x[i] - x[i-1]);
            dxy[2] = (y[i] - y[i-1]);
        } else {
            dxy[1] = (x[i+1] - x[i-1]);
            dxy[2] = (y[i+1] - y[i-1]);
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

		// the x locations of pixels
		vector[n_pix] x;
		// the y locations of pixels
		vector[n_pix] y;

		// ----------------------------------------------------------
		// Stream nodes:

		// number of nodes
		int n_nodes;

		// nodes locations along rigid polynomial
		vector[n_nodes] phi1_nodes;
		vector[n_nodes] phi2_nodes_init;

		// width of nodes: along rigid polynomial, h
		vector[n_nodes] h_nodes;
}

parameters {
    // Stream model:
	vector<lower=-0.3, upper=0.3>[n_nodes] d_phi2_nodes;
	vector<lower=-1.8, upper=0.>[n_nodes] log_w_nodes;
	vector<lower=-8, upper=8>[n_nodes] log_a_nodes;
}

transformed parameters {
	vector[n_pix] log_bg_int;
	vector[n_pix] log_gd1_int;
	vector[n_pix] xmod;

	vector[n_nodes] tmp;
	vector[2] xy;
	vector[2] node_xy;

	vector[n_nodes] phi2_nodes;

    matrix[2,2] R[n_nodes];

	// Un-log some things
	vector[n_nodes] w_nodes;
    matrix[2,2] C[n_nodes];

	w_nodes = exp(log_w_nodes);
	phi2_nodes = phi2_nodes_init + d_phi2_nodes;

    // Re-compute node rotation matrices
    for (i in 1:n_nodes) {
        R[i] = get_R(i, n_nodes, phi1_nodes, phi2_nodes);
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
		xmod[i] = log_gd1_int[i];
	  }
}
model {
    // Priors
    for (n in 1:n_nodes) {
        log_w_nodes[n] ~ normal(-1, 0.2)T[-1.8, 0.];
        d_phi2_nodes[n] ~ normal(0, 0.1)T[-0.3, 0.3];
    }

    //Likelihood
    hh ~ poisson_log(xmod);
}
