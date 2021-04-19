library(R6)
library(inline)
library(RcppEigen)


#### C++ code

forward_cpp = cxxfunction(
  signature(P_r="NumericMatrix", delta_r="NumericVector", Gamma_r="NumericMatrix"),
  plugin="RcppEigen",
  body="
    using Eigen::Map;
    using Eigen::VectorXd;
    using Eigen::MatrixXd;
    
    const Map<MatrixXd> P(as<Map<MatrixXd>>(P_r));
    const Map<VectorXd> delta(as<Map<VectorXd>>(delta_r));
    const Map<MatrixXd> Gamma(as<Map<MatrixXd>>(Gamma_r));
    
    VectorXd forward(P.col(0).cwiseProduct(delta));
    double sum_forward = forward.sum();
    if(sum_forward == 0) return wrap(std::numeric_limits<double>::infinity());
    double ll = log(sum_forward);
    VectorXd phi(forward / sum_forward);
    
    for(int t = 1; t < P.cols(); t++){
      forward = P.col(t).cwiseProduct(Gamma.transpose() * phi);
      sum_forward = forward.sum();
      if(sum_forward == 0) return wrap(std::numeric_limits<double>::infinity());
      ll += log(sum_forward);
      phi = forward / sum_forward;
    }
    
    return wrap(-ll);
")


#### Model class

Hmm = R6Class(
  "Hmm",
  public=list(
    # Fields
    nr_states = NULL,
    delta     = NULL,
    gamma     = NULL,
    alphas    = NULL,
    betas     = NULL,
    
    # Methods
    initialize = function(nr_states){
      nr_states <<- nr_states
    },
    
    get_means = function(){
      return(alphas / betas)
    },
    
    get_vars = function(){
      return(alphas / betas^2)
    },
    
    theta_to_params = function(theta){
      #' translate optimization vector theta to parameters of the model
      #' 
      #' The diagonal elements of gamma are fixed at 1 and the rest is filled by the first
      #' values of theta, a softmax is done to get probabilities.
      #' Delta is determined by gamma.
      #' The rest of theta stores the alphas and the betas, we use the exponential
      #' function to make sure the values are positive.
      
      i = 1
      j = (nr_states - 1) * nr_states
      gamma <<- diag(nr_states)
      gamma[!gamma] <<- exp(theta[i:j])
      gamma <<- gamma / rowSums(gamma)
      
      delta <<- solve(t(diag(nr_states) - gamma + 1), rep(1, nr_states))
      
      i = j + 1
      j = i + nr_states - 1
      alphas <<- exp(theta[i:j])
      
      i = j + 1
      j = i + nr_states - 1
      betas <<- exp(theta[i:j])
    },
    
    sort_components = function(order=NULL){
      #' Reorder the components (states) by their share of the static distribution
      #' and delta and gamma accordingly
      
      if (is.null(order)) {
        i_sorted = sort(delta, decreasing=TRUE, index.return=TRUE)$ix
      } else {
        i_sorted = order
      }
      
      
      permut = matrix(0, nrow=nr_states, ncol=nr_states)  # permutation matrix
      for (i in 1:nr_states) {
        permut[i,i_sorted[i]] = 1
      }
      
      delta  <<- c(delta %*% t(permut))
      gamma  <<- permut %*% gamma %*% t(permut)
      alphas <<- c(alphas %*% t(permut))
      betas  <<- c(betas %*% t(permut))
    },
    
    get_initial_theta = function(x){
      g = rep(-2.0, (nr_states - 1) * nr_states)
      
      ## Method 1: equal initialisation + some randomness
      # add some randomnes (gamma dist with mean mu and very small variance 0.00002)
      #mus = rgamma(nr_states, shape=500, rate=500/mean(x))
      # for some reason results are more stable when we start with mu slightly higher than data mean
      # it's possible that this is just avoiding a local optimum specific to our test data
      #mus = mus * 1.2
      # this will be the approximate mean var of the components after training
      #sigma2 = var(x) / nr_states
      
      ## Method 2: kmeans
      res = kmeans(x, nr_states, nstart=5)
      mus = res$centers
      # computes within cluster-variance from within-cluster sum of squares
      sigma2 = sapply(1:nr_states, function(i){res$withinss[i] / sum(res$cluster == i)})
      
      # IDEE von roland quantile(VeDBA,c(runif(1,0,0.2),runif(1,.0.3,0.6),runif(1,0.7,0.9))) 
      # für jedes quantil ein mu
      
      
      a = mus^2 / sigma2
      b = mus / sigma2
      return(c(g, log(a), log(b)))
    },
    
    emission_pdfs = function(x){
      # our method of optimizing likelihood does not work with 0 data
      epsilon = 1E-10
      x[x == 0] = epsilon
      
      p = matrix(1, length(x), nr_states)
      for (k in 1:nr_states) {
        p[, k] = dgamma(x, shape=alphas[k], rate=betas[k])
      }
      return(p)
    },
    
    emission_cdfs = function(x){
      epsilon = 1E-10
      x[x == 0] = epsilon
      
      p = matrix(1, length(x), nr_states)
      for (k in 1:nr_states) {
        p[, k] = pgamma(x, shape=alphas[k], rate=betas[k])
      }
      return(p)
    },
    
    marginal_probability = function(x){
      return(emission_pdfs(x) %*% delta)
    },
    
    nll = function(x){
      allprobs = emission_pdfs(x)
      
      forward = delta * allprobs[1,]
      sum_forward = sum(forward)
      if (sum_forward == 0) return(Inf)
      ll = log(sum_forward)
      phi = forward / sum_forward
      for (t in 2:length(x)) {
        forward = phi %*% gamma * allprobs[t,]
        sum_forward = sum(forward)
        if (sum_forward == 0) return(Inf)
        ll = ll + log(sum_forward)
        phi = forward / sum_forward
      }
      return(-ll)
    },
    
    nll_cpp = function(x){
      allprobs = emission_pdfs(x)
      return(forward_cpp(t(allprobs), delta, gamma))
    },
    
    fit = function(x, ...){
      theta_0 = get_initial_theta(x)
      
      objective = function(theta){
        theta_to_params(theta)
        #return(nll(x))
        return(nll_cpp(x))
      }
      
      # optim-BFGS, optim-CG do not work (immediately explodes to unreasonable values)
      # nlm only works with specified step size
      # save hessian for confidence interval estimation
      res = nlm(objective, theta_0, stepmax=5, ...)
      theta_to_params(res$estimate)
      
      #res = optim(theta_0, objective, method="L-BFGS-B")
      #res = optim(theta_0, objective, method="Nelder-Mead", control=list(maxit=2000))
      #theta_to_params(res$par)
      
      #res = nlminb(theta_0, objective)
      #theta_to_params(res$par)
      
      sort_components()
      
      return(res)
    },
    
    entropy = function(x){
      #' Calculate entropy of the hidden state sequence given the model and emissions x
      #' Using an algorithm proposed by Hernando et al 2005
      #' 
      #' entropy = 0 if there is only one possible state sequence that could have produced x
      #' entropy = length(x) * log(nr_states) if all state sequences have the same probability
      #' given the model
      #' The higher the entropy the more uncertainty the model has about its prediction
      
      allprobs = emission_pdfs(x)
      
      h = rep(0, nr_states)
      c = delta * allprobs[1,]
      c = c / sum(c)
      for (t in 2:length(x)) {
        p = sweep(gamma, MARGIN=1, c, FUN='*')
        p = sweep(p, MARGIN=2, colSums(p), FUN="/")  # normalize columns
        plogp = p * log(p)
        plogp[is.na(plogp)] = 0  # this sets p*log(p)=0 for p=0 without producing NaN
        h = h %*% p - colSums(plogp)
        c = c %*% gamma * allprobs[t,]
        c = c / sum(c)
      }
      # this can still lead to NaN sometimes for big numbers of states due to one entry
      # of c being equal to zero
      return((c %*% t(h - log(c)))[1])
    },
    
    predict_states = function(x){
      #' Viterbi algorithm
      
      allprobs = emission_pdfs(x)
      
      xi = matrix(0, length(x), nr_states)
      xi[1,] = delta * allprobs[1,]
      xi[1,] = xi[1,] / sum(xi[1,])
      for (t in 2:length(x)){
        xi[t,] = apply(xi[t-1,] * gamma, 2, max) * allprobs[t,]
        xi[t,] = xi[t,] / sum(xi[t,])
      }
      iv = numeric(length(x))
      iv[length(x)] = which.max(xi[length(x),])
      for (t in (length(x)-1):1){
        iv[t] = which.max(gamma[,iv[t+1]] * xi[t,])
      }
      return(iv)
    },
    
    sample_from_model = function(n){
      states = 1:nr_states
      sampled_states = numeric(n)
      sampled_states[1] = sample(states, 1, prob=delta)
      for (t in 2:n){
        sampled_states[t] = sample(states, 1, prob=gamma[sampled_states[t-1],])
      }
      sampled_emissions = rgamma(n, shape=alphas[sampled_states], rate=betas[sampled_states])
      return(list(states=sampled_states, emission=sampled_emissions))
    },
    
    log_forward = function(x){
      allprobs = emission_pdfs(x)
      
      alpha     = matrix(NA, nr_states, length(x))
      phi       = delta * allprobs[1,]
      sum_phi   = sum(phi)
      ll        = log(sum_phi)
      phi       = phi / sum_phi
      alpha[,1] = ll + log(phi)
      for (t in 2:length(x)) {
        phi       = phi %*% gamma * allprobs[t,]
        sum_phi   = sum(phi)
        ll        = ll + log(sum_phi)
        phi       = phi / sum_phi
        alpha[,t] = log(phi) + ll
      }
      return(alpha)
    },
    
    log_backward = function(x){
      allprobs = emission_pdfs(x)
      
      beta             = matrix(NA, nr_states, length(x))
      beta[,length(x)] = rep(0, nr_states)
      phi              = rep(1/nr_states, nr_states)
      ll               = log(nr_states)
      for (t in (length(x)-1):1) {
        phi      = gamma %*% (allprobs[t+1,] * phi)
        beta[,t] = log(phi) + ll
        sum_phi  = sum(phi)
        phi      = phi / sum_phi
        ll       = ll + log(sum_phi)
      }
      return(beta)
    },
    
    cond_dist = function(x){
      #' P(Xt < xt | X1=x1,...,Xt-1=xt-1,Xt+1=xt+1,...,XT=xT)
      e = emission_cdfs(x)
      la = log_forward(x)
      lb = log_backward(x)
      # scale forward and backward such that exp won't overflow
      la = sweep(la, MARGIN=2, apply(la, MARGIN=2, max), '-')
      lb = sweep(lb, MARGIN=2, apply(lb, MARGIN=2, max), '-')
      a = exp(la)
      b = exp(lb)
      a = cbind(delta, a[,-length(x)])
      p_states = apply(a, MARGIN=2, function(ai){return(ai %*% gamma)}) * b
      p_states = sweep(p_states, MARGIN=2, colSums(p_states), '/')
      return(colSums(p_states*t(e)))
    },
    
    pseudo_residuals = function(x){
      return(qnorm(cond_dist(x)))
    }
  ),
  
  # for performance boost
  portable = FALSE, 
  class = FALSE,
  cloneable = FALSE
)


