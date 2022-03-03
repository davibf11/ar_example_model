library(tidyverse)
library(rstan)
options(stringsAsFactors = F, digits = 5, mc.cores = 4)

####Simulate data

num_per_group <- 20
max_time <- 5
num_group <- 150
sigma <- 5
sd_1 <- 0.5
Kar <- 1
phi_1 <- 0.7

####Generate sqrt AR(1) cor matrix

sigma_phi = (sd_1 ^ 2) / (1 - phi_1^2)

r <- c()
r[1] = 1
for(i in 2:max_time) {
  r[i] = phi_1 * r[i - 1]
}

V <- matrix(rep(0, max_time^2), nrow = max_time)

for(i in 1:max_time) {
  for(j in 1:max_time) {
    V[i, j] = sigma_phi * r[abs(i - j) + 1]
  }
}

eigenvals = eigen(V)$values
eigen_sqrt <- c()
for(i in 1:max_time) {
  eigen_sqrt[i] = sqrt(eigenvals[i]);
}
eigenvec_matrix <- eigen(V)$vectors

V_sqrt = eigenvec_matrix %*% diag(eigen_sqrt) %*% t(eigenvec_matrix)

##generate group means
set.seed(12)
group_means_pre_transform <- matrix(rnorm(num_group*max_time, mean = 0, sd = 1), nrow = num_group)
group_means <- group_means_pre_transform %*% V_sqrt

##generate actual data
N <- num_group*max_time*num_per_group
time_base <- c()
for(i in 1:max_time) {
  time_base <- c(time_base, rep(i, num_per_group))
}
time <- rep(time_base, num_group)
individual <- c()
for(i in 1:num_group) {
  individual <- c(individual, rep(i, max_time * num_per_group))
}
weights <- rpois(N, lambda = 25)
Y <- c()
for(i in 1:num_group) {
  for(j in 1:max_time) {
    for(k in 1:num_per_group) {
      Y <- c(Y, rnorm(1, mean = group_means[i,j], sd = sigma/sqrt(weights[i])))
    }
  }
}

##Add global intercept
Y <- Y + 7

##Run Stan model

## Can optionally use VI first to set initial values and improve runtime

# vb_model <- stan_model(file = "/ar_example_model/ar_RE_example.stan")
# 
# vb_fit <- vb(object = vb_model,
#              data = list(
#                N = N,
#                Y = Y,
#                weights = weights,
#                max_time = max_time,
#                Kar = Kar,
#                time = time,
#                N_1 = num_group,
#                M_1 = 1,
#                J_1 = individual,
#                prior_only = FALSE
#              ),
#              iter = 10000,
#              seed = 12,
#              output_samples = 1000
# )
# 
# vb_draws_df <- posterior::as_draws_df(vb_fit)
# 
# init_list <- split(vb_draws_df[1:4 * floor(1000 / 4),], seq(4))

fit <- stan(file = "/ar_example_model/ar_RE_example.stan",
            data = list(
              N = N,
              Y = Y,
              weights = weights,
              max_time = max_time,
              Kar = Kar,
              time = time,
              N_1 = num_group,
              M_1 = 1,
              J_1 = individual,
              prior_only = FALSE
            ),
            init = init_list,
            warmup = 2500, iter = 5000,
            chains = 4,
            cores = 4,
            thin = 1,
            seed = 1492,
            control = list(max_treedepth = 20, adapt_delta = 0.975),
            verbose = F
)
