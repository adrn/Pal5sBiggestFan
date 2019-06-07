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

    // get the vector of spacings between nodes
    vector geths(int n_nodes, vector nodes)
    {
      int n = n_nodes -1;
      vector[n] hs;
      for (i in 1:n)
      {
        hs[i] = nodes[i+1]-nodes[i];
      }
      return hs;
    }
    // obtain the vector of spline coefficients given the location
    // of the nodes and values there
    // We are using natural spline definition
    vector getcoeffs(int n_nodes, vector nodes, vector vals)
    {
      int n=n_nodes-1;
      vector[n] hi;
      vector[n] bi;
      vector[n-1] vi;
      vector[n-1] ui;
      vector[n_nodes] ret;
      vector[n-1] zs;
      matrix[n-1,n-1] M = rep_matrix(0, n-1, n-1);

      n = n_nodes-1;

      for (i in 1:n)
      {
        hi[i] = nodes[i+1]-nodes[i];
        bi[i] =  1/hi[i]*(vals[i+1]-vals[i]);
      }
      for (i in 2:n)
      {
        vi[i-1] = 2*(hi[i-1]+hi[i]);
        ui[i-1] = 6*(bi[i] - bi[i-1]);
      }
      for (i in 1:n-1)
      {
        M[i,i] = vi[i];
      }
      for (i in 1:n-2)
      {
        M[i+1,i] = hi[i];
        M[i,i+1] = hi[i];
      }
      //print (M)
      zs = M \ ui ; //mdivide_left_spd(M, ui);
      ret[1]=0;
      ret[n_nodes] =0;
      ret[2:n_nodes-1]=zs;

      return ret;

    }

    // Evaluate the spline, given nodes, values at the nodes
    // spline coefficients, locations of evaluation points
    // and integer bin ids of each point
    vector spline_eval(int n_nodes, vector nodes,
           vector vals, vector zs,
           int n_dat, vector x, int[] i)
    {

      vector[n_nodes-1] h;
      vector[n_dat] ret;
      int i1[n_dat];
      for (ii in 1:n_dat)
      {
        i1[ii] = i[ii] + 1;
      }
      h = geths(n_nodes, nodes);

      ret = (
          zs[i1] ./ 6 ./ h[i] .* square(x-nodes[i]) .*(x-nodes[i])+
          zs[i]  ./ 6 ./ h[i] .* square(nodes[i1]-x) .* (nodes[i1]-x)+
          (vals[i1] ./ h[i] - h[i] .* zs[i1] ./ 6) .* (x-nodes[i])+
          (vals[i] ./ h[i] - h[i] .* zs[i] ./ 6) .* (nodes[i1]-x)
          );
      return ret;
    }

    // find in which node interval we should place each point of the vector
    int[] findpos(int n_nodes, vector nodes, int n_dat, vector x)
    {
      int ret[n_dat];
      for (i in 1:n_dat)
      {
        for (j in 1:n_nodes-1)
        {
          if ((x[i]>=nodes[j]) && (x[i]<nodes[j+1]))
          {
            ret[i] = j;
          } else {
          }
        }
      }
      return ret;
    }
}

data {
		// number of pixels in the density map
		int n_pix;
		// number counts of objects in pixels
		int hh[n_pix];

		// selection function for each pixel
		real S[n_pix];

		// the x locations of pixels
		vector[n_pix] x;
		// the y locations of pixels
		vector[n_pix] y;

		// ----------------------------------------------------------
		// Nodes:

		// number of nodes
		int n_nodes;

		// nodes locations along rigid polynomial
		vector[n_nodes] phi1_nodes;
		vector[n_nodes] phi2_nodes_init;

		// width of nodes: along rigid polynomial, h
		vector[n_nodes] h_nodes;

    // Nodes for the background spline:
    int n_bg_nodes; // number of nodes
    vector[n_bg_nodes] bg_nodes; // phi1 locations of background nodes

}

transformed data {
		real log_S[n_pix] = log(S); // selection function
    matrix[2,2] R[n_nodes];

    int node_ids_bg[n_pix] = findpos(n_bg_nodes, bg_nodes, n_pix, x);

    // pre-compute rotation matrices
    for (i in 1:n_nodes) {
        R[i] = get_R(i, n_nodes, phi1_nodes, phi2_nodes_init);
    }
}

parameters {
		vector<lower=-0.3, upper=0.3>[n_nodes] d_phi2_nodes;
		vector<lower=-2, upper=-0.5>[n_nodes] log_w_nodes;
		vector<lower=-8, upper=8>[n_nodes] log_a_nodes;
    vector<lower=-10, upper=4>[n_bg_nodes] log_bg_nodes;
}

transformed parameters {
		vector[n_pix] log_bg_int;
		vector[n_pix] log_gd1_int;
		vector[n_pix] xmod;

		vector[n_nodes] tmp;
		vector[2] xy;
		vector[2] node_xy;

		vector[n_nodes] phi2_nodes;

		real ln_bg_val;

		// Un-log some things
		vector[n_nodes] w_nodes;
		vector[n_nodes] a_nodes;

    matrix[2,2] C[n_nodes];

    // Background spline
    vector[n_bg_nodes] coeffs_bg = getcoeffs(n_bg_nodes, bg_nodes, log_bg_nodes);
    log_bg_int = spline_eval(n_bg_nodes, bg_nodes, log_bg_nodes, coeffs_bg,
  	                         n_pix, x, node_ids_bg);

		w_nodes = exp(log_w_nodes);

		phi2_nodes = phi2_nodes_init + d_phi2_nodes;

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
				xmod[i] = log_sum_exp(log_gd1_int[i], log_bg_int[i]) + log_S[i];
        // print("gd1_int ", log_gd1_int[i], " xmod ", xmod[i], " log_bg_int ", log_bg_int[i], " log_S ", log_S[i]);
	  }
}
model {
    // Priors
    for (n in 1:n_nodes) {
        log_w_nodes[n] ~ normal(log(0.15), 0.35)T[-2, -0.5];
        d_phi2_nodes[n] ~ normal(0, 0.1)T[-0.3, 0.3];
    }
    // target += -log_diff_exp(normal_lcdf(-0.5| log(0.15), 0.35),
    //                         normal_lcdf(-2| log(0.15), 0.35));
    target += -log_a_nodes; // like a regularization term to force amplitudes to 0

    //Likelihood
    hh ~ poisson_log(xmod);
}
