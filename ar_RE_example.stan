functions {
      
  matrix create_sqrt_cov_matrix_ar1(int N, real sigma, real phi) {
  	matrix[N, N] V;
  	matrix[N, N] v_sqrt;
  	vector[N] eigenvals;
  	vector[N] r;
	  real sigma_phi;
    
    sigma_phi = (sigma ^ 2) / (1 - phi^2);
    
	  r[1] = 1;
	  for(i in 2:N) {
	  	r[i] = phi * r[i - 1];
  	}
  	
  	for(i in 1:N) {
  	  for(j in 1:N) {
  		  V[i, j] = sigma_phi * r[abs(i - j) + 1]; 
  	  }
  	}
	  
	  eigenvals = eigenvalues_sym(V);
	  for(i in 1:N) {
  	  eigenvals[i] = sqrt(eigenvals[i]);
  	}
  	
  	v_sqrt = eigenvectors_sym(V) * diag_matrix(eigenvals) * eigenvectors_sym(V)';
  	
  	return v_sqrt;
  }
  
  matrix create_sqrt_cov_matrix_ar2(int N, real sigma, real phi1, real phi2) {
    matrix[N, N] V;
    matrix[N, N] v_sqrt;
    vector[N] eigenvals;
    vector[N] r;
	  real sigma_phi;
	
	  sigma_phi = ((1 - phi2) * (sigma ^ 2)) / ((1 + phi2) * ((1 - phi2) ^ 2 - phi1 ^ 2));
    
    r[1] = 1;
    r[2] = phi1 / (1 - phi2);
    for (i in 3:N) {
      r[i] = phi1 * r[i-1] + phi2 * r[i-2];
    }
    
    for(i in 1:N) {
      for(j in 1:N) {
        V[i, j] = sigma_phi * r[1+abs(i - j)];
      }
    }
    
    eigenvals = eigenvalues_sym(V);
    for(i in 1:N) {
      eigenvals[i] = sqrt(eigenvals[i]);
    }
    
    v_sqrt = eigenvectors_sym(V) * diag_matrix(eigenvals) * eigenvectors_sym(V)';
    
    return v_sqrt;
  }
  
  matrix create_sqrt_cov_matrix_ar3(int N, real sigma, real phi1, real phi2, real phi3) {
	matrix[N, N] V;
	matrix[N, N] v_sqrt;
	vector[N] eigenvals;
	vector[N] r;
	real sigma_phi;
	
	sigma_phi = (sigma^2) / (1 - (phi1^2 + phi2^2 + phi3^2 + 2*r[2]*(phi1*phi2 + phi2*phi3) + 2*r[3]*phi1*phi3));
	
	r[1] = 1; // Yule-Walker Equations solutions
	r[2] = (phi1 + phi2 * phi3) / (1 - phi2 - phi3 * (phi1 + phi3));
	r[3] = (phi1 + phi3)*r[2] + phi2;
	
	if(N > 3) {
	  for(i in 4:N) {
		  r[i] = phi1 * r[i - 1] + phi2 * r[i - 2] + phi3 * r[i - 3];
	  }
	}
	
	for(i in 1:N) {
	  for(j in 1:N) {
		  V[i, j] = sigma_phi * r[abs(i - j) + 1]; 
	  }
	}
	
	eigenvals = eigenvalues_sym(V);
	for(i in 1:N) {
	  eigenvals[i] = sqrt(eigenvals[i]);
	}
	
	v_sqrt = eigenvectors_sym(V) * diag_matrix(eigenvals) * eigenvectors_sym(V)';
	
	return v_sqrt;
  }
  
  matrix ar1_transform(matrix X, int M, int N, real sigma, real phi) {
    matrix[M, N] X_hat;
    
    X_hat = X * create_sqrt_cov_matrix_ar1(N, sigma, phi); 
    return X_hat;
  }
  
  matrix ar2_transform(matrix X, int M, int N, real sigma, real phi_1, real phi_2) {
    matrix[M, N] X_hat;
    
    X_hat = X * create_sqrt_cov_matrix_ar2(N, sigma, phi_1, phi_2); 
    return X_hat;
  }
  
  matrix ar3_transform(matrix X, int M, int N, real sigma, real phi_1, real phi_2, real phi_3) {
    matrix[M, N] X_hat;
    
    X_hat = X * create_sqrt_cov_matrix_ar3(N, sigma, phi_1, phi_2, phi_3); 
    return X_hat;
  }
}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  vector<lower=0>[N] weights;  // model weights
  // data needed for AR correlations
  int<lower=1> max_time;  // maximum value of time vector
  int<lower=0> Kar;  // AR order
  int<lower=1> time[N]; // time component
  // data for group-level effects of ID 1
  int<lower=1> N_1;  // number of grouping levels
  int<lower=1> M_1;  // number of coefficients per level
  int<lower=1> J_1[N];  // grouping indicator per observation
  int prior_only;  // should the likelihood be ignored?
}
parameters {
  real Intercept;  // temporary intercept for centered predictors
  real<lower=-1,upper=1> ar[Kar];  // autoregressive coefficients
  real<lower=0> sigma;  // residual SD
  vector<lower=0>[M_1] sd_1;  // group-level standard deviations
  matrix[N_1, max_time] z_1;  // standardized group-level effects
}
transformed parameters {
  matrix[N_1, max_time] r_1_1;  // actual group-level effects
  if(Kar == 1) {
    r_1_1 = ar1_transform(z_1, N_1, max_time, sd_1[1], ar[1]);
  } else if(Kar == 2) {
    r_1_1 = ar2_transform(z_1, N_1, max_time, sd_1[1], ar[1], ar[2]);
  } else if(Kar == 3) {
    r_1_1 = ar3_transform(z_1, N_1, max_time, sd_1[1], ar[1], ar[2], ar[3]);
  }
} 
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = Intercept + rep_vector(0.0, N);
    for (n in 1:N) {
      // add more terms to the linear predictor
      mu[n] += r_1_1[J_1[n], time[n]];
    }
    for (n in 1:N) {
      target += weights[n] * (normal_lpdf(Y[n] | mu[n], sigma));
    }
  }
  // priors including constants
  target += student_t_lpdf(Intercept | 3, 0.3, 2.5);
  target += student_t_lpdf(sigma | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  target += student_t_lpdf(sd_1 | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  target += std_normal_lpdf(to_vector(z_1));
  target += uniform_lpdf(ar | -1, 1);
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept;
}
