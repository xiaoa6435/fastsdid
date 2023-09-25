use extendr_api::prelude::*;

fn fw_step<'a>(
    x: ArrayView2<f64>, y: ArrayView1<f64>,
    mut w: ArrayViewMut1<'a, f64>,
    eta: f64, fit_intercept: bool
) -> (ArrayViewMut1<'a, f64>, Array<f64, Ix1>, f64) {

  let (ysum, col_means) = if fit_intercept {
    let ysum = y.sum();
    let col_means = x.mean_axis(Axis(0)).unwrap();
    (ysum, col_means)
  } else {
    (0.0, Array::zeros(0))
  };
  let d = if fit_intercept {(col_means).dot(&w)} else {0.0};
  let xw = &x.dot(&w) - d;

  let half_grad = if fit_intercept {
    (&xw - &y).t().dot(&x) + eta * &w + ysum * &col_means
  } else {
    (&xw - &y).t().dot(&x) + eta * &w 
  };

  // which.min
  let i = half_grad.iter().enumerate().
      fold(
          (usize::MAX, f64::MAX), 
          |min, (ind, &val)| if val < min.1 {(ind, val)} else {min}
      ).0;
  let mut dw = -&w;
  dw[i] = 1.0 - w[i];

  let d_err_intercept = if fit_intercept {col_means[i]} else {0.0};
  let d_err = &(x.index_axis(Axis(1), i)) - &xw - d_err_intercept;
  let step = -(&half_grad).dot(&dw) / (&d_err.dot(&d_err) + eta * &dw.dot(&dw));
  let constrained_step = f64::max(f64::min(1.0, step), 0.0);
  w += &(constrained_step * &dw);

  let err_intercept = if fit_intercept {
    col_means.dot(&w) - ysum / (y.len() as f64)
  } else {
    0.0
  };

  let err = (&x).dot(&w) - &y - err_intercept;
  let vals = eta * eta * (&w).dot(&w) + (&err).dot(&err) / (err.len() as f64);
  //rprintln!("vals:{}, w:{:?}, err:{:?}, {}, {:?}", vals, w.to_vec(), err.to_vec(), i, half_grad.to_vec());
  (w, err, vals)
}

#[extendr]
fn fw_sc_weight(
  xy: ArrayView2<f64>, w: ArrayView1<f64>, fit_intercept: bool, eta: f64, min_decrease: f64, max_iter: i32
) -> List {
  let n = xy.shape()[1];
  assert!(w.len() == n - 1);
  let x = xy.slice_axis(Axis(1), Slice::from(0..(n - 1)));
  let y = xy.index_axis(Axis(1), n - 1);
  let min_decrease_sqr = min_decrease * min_decrease;
  let mut t = 0;
  let mut vals_last = f64::MAX;
  let mut w = w.to_owned();
  while t < max_iter {
    let vals = fw_step(x, y, w.view_mut(), eta, fit_intercept).2;
    //rprintln!("t:{},s:{},w:{:?}", t + 1, vals, w.to_vec());
    if vals_last - vals <= min_decrease_sqr {
      break
    };
    vals_last = vals;
    t += 1;
  }
  list!(lambda = w.to_vec())
}

fn update_w(
  w: ArrayViewMut1<f64>, eta: f64, fit_intercept: bool,
  mut beta: ArrayViewMut1<f64>,
  y_beta: ArrayView2<f64>,
  cov: ArrayView3<f64>, alpha: f64,
  axis: usize
) -> f64 {
  let _xy = y_beta.slice_axis(Axis(axis), Slice::from(0..(y_beta.shape()[axis] - 1)));
  let xy = if axis == 0 {_xy} else {_xy.t()};
  let (m, n) = (xy.shape()[0], xy.shape()[1]);
  assert!(w.len() == n - 1);
  let x = xy.slice_axis(Axis(1), Slice::from(0..(n - 1)));
  let y = xy.index_axis(Axis(1), n - 1);
  let (w, err, vals) = fw_step(x, y, w, eta, fit_intercept);
  if cov.len() > 0 {
    let grad_beta = cov
      .slice_axis(Axis(axis), Slice::from(0..m))
      .map_axis(Axis(axis), |mv| (&err).dot(&mv))
      .map_axis(Axis(0), |nv| {
        let init = nv.slice_axis(Axis(0), Slice::from(0..(n - 1)));
        -(init.dot(&w) - nv[n - 1]) / (m as f64)
      });
    beta -= &(grad_beta * alpha);
  }
  vals
}

#[extendr]
fn fw_sc_weight_covariates(
  xty: ArrayView2<f64>, cov: RMatrix3D<f64>, beta: ArrayView1<f64>,
  lambda: ArrayView1<f64>, eta_lambda: f64, fit_intercept_lambda: bool, update_lambda: bool,
  omega: ArrayView1<f64>, eta_omega: f64, fit_intercept_omega: bool, update_omega: bool,
  min_decrease: f64, max_iter: i32
) -> List {
  
  assert!(
    cov.len() == 0 || (xty.shape()[0] == cov.dim()[0] && xty.shape()[1] == cov.dim()[1]),
    "xty.shape:{:?}, cov.dim:{:?}", xty.shape(), cov.dim()
  );
  let cov = Array::from_shape_vec(cov.dim().f(), cov.data().to_vec()).unwrap();
  let cov = cov.view();
  
  let mut t = 0;
  let mut vals = 0.0;
  let mut vals_last = f64::MAX;
  let mut y_beta = xty.to_owned();
  let mut beta = beta.to_owned();
  let mut new_beta = beta.clone();
  let mut lambda = lambda.to_owned();
  let mut omega = omega.to_owned();
  let min_decrease_sqr = min_decrease * min_decrease;
  let update_beta = cov.len() > 0;
  while t < max_iter + 1 && (t < 3 || vals_last - vals > min_decrease_sqr) {
    beta = new_beta.clone();
    rprintln!("t:{},s:{},b:{:?},lambda:{:?},omega:{:?}", t, vals, beta.to_vec(), lambda.to_vec(), omega.to_vec());
    if update_beta {
      y_beta = &xty - &(&cov).map_axis(Axis(2), |m| (&m).dot(&beta));
    }
    let alpha = 1.0 / ((t + 1) as f64);
    vals_last = vals;
    vals = 0.0;
    if update_lambda {
      vals += update_w(
        lambda.view_mut(), eta_lambda, fit_intercept_lambda, 
        new_beta.view_mut(), y_beta.view(), cov, alpha, 0
      )
    }
    if update_omega {
      vals += update_w(
        omega.view_mut(), eta_omega, fit_intercept_omega,
        new_beta.view_mut(), y_beta.view(), cov, alpha, 1
      )
    }
    t += 1;
  }
  list!(lambda = lambda.to_vec(), omega = omega.to_vec(), beta = beta.to_vec()) 
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod fastsdid;
    fn fw_sc_weight;
    fn fw_sc_weight_covariates;
}

/*
// library(synthdid)
library(tidyverse)

rextendr::document()
devtools::load_all(".")

data('california_prop99')
setup = panel.matrices(california_prop99)
tau.hat = synthdid_estimate(setup$Y, setup$N0, setup$T0)
tau.hat
synthdid::synthdid_estimate(setup$Y, setup$N0, setup$T0)
// synthdid: -15.604 +- NA. Effective N0/N0 = 16.4/38~0.4. Effective T0/T0 = 2.8/19~0.1. N1,T1 = 1,12.
// synthdid: -10.526 +- NA. Effective N0/N0 = 4.0/38~0.1. Effective T0/T0 = 2.8/19~0.1. N1,T1 = 1,12. 

v2s <- function(v) str_c(v, collapse = ', ')
fw.step = function(A, x, b, eta) {
  Ax = A %*% x
  half.grad = t(Ax - b) %*% A + eta * x
  i = which.min(half.grad)
  
  d.x = -x
  d.x[i] = 1 - x[i]
  if (all(d.x == 0)) { return(x) }
  d.err = A[, i] - Ax
  step = -t(c(half.grad)) %*% d.x / (sum(d.err^2) + eta * sum(d.x^2))
  constrained.step = min(1, max(0, step))
  # return(x + constrained.step * d.x)
  
  x = x + constrained.step * d.x
  err = A %*% x - b
  vals = Re(eta^2) * sum(x^2) + sum(err^2) / length(err)
  # print(glue::glue("vals:{vals}, w:[{v2s(x)}], err:[{v2s(err)}], {i}, [{v2s(half.grad)}]"))
  x
}

// set.seed(0)
// m = 10
// n = 3
// x = matrix(rnorm(m * n), m, n)
// y = matrix(rnorm(m * 1), m, 1)
// w = rep(1 / n, n)
// eta = 1e-6
// fit_intercept = FALSE
// fw_step(x, y, w, eta, fit_intercept)
// fw.step(x, w, y, eta)

# a Frank-Wolfe step for \\Ax - b||^2 + eta * ||x||^2 with x in unit simplex.

# a Frank-Wolfe solver for synthetic control weights using exact line search
sc.weight.fw = function(Y, zeta, intercept = TRUE, lambda = NULL, min.decrease = 1e-3, max.iter = 1000) {
  T0 = ncol(Y) - 1
  N0 = nrow(Y)
  if (is.null(lambda)) { lambda = rep(1 / T0, T0) }
  if (intercept) {
    Y = apply(Y, 2, function(col) { col - mean(col) })
  }

  t = 0
  vals = rep(NA, max.iter)
  A = Y[, 1:T0]
  b = Y[, T0 + 1]
  eta = N0 * Re(zeta^2)
  while (t < max.iter && (t < 2 || vals[t - 1] - vals[t] > min.decrease^2)) {
    t = t + 1
    lambda.p = fw.step(A, lambda, b, eta)
    lambda = lambda.p
    err = Y[1:N0, ] %*% c(lambda, -1)
    vals[t] = Re(zeta^2) * sum(lambda^2) + sum(err^2) / N0
    # print(glue::glue("t:{t},s:{vals[t]},w:[{v2s(lambda)}]"))
    
  }
  list(lambda = lambda, vals = vals)
}

set.seed(0)
m = 10
n = 4
x = matrix(rnorm(m * n), m, n)
y = matrix(rnorm(m * 1), m, 1)
xy = cbind(x, y)
eta = 1e-6
fit_intercept = TRUE
lambda = rep(1 / n, n)
min_decrease = 1e-3
max_iter = 100000

sc.weight.fw.v2 = function(Y, zeta, intercept = TRUE, lambda = NULL, min.decrease = 1e-3, max.iter = 1000) { # nolint # nolint
    T0 = ncol(Y) - 1
    if (is.null(lambda)) { lambda = rep(1 / T0, T0) }
    fw_sc_weight(Y, lambda, intercept, zeta, min.decrease, max.iter)
    # fw_sc_weight(Y)
}
old = sc.weight.fw(xy, eta, fit_intercept, lambda, min_decrease, max_iter)$lambda
new_v2 = sc.weight.fw.v2(xy, eta, fit_intercept, lambda, min_decrease, max_iter)$lambda
new = fw_sc_weight(xy, lambda, fit_intercept, eta, min_decrease, max_iter)$lambda
all(abs(old - new) < 1e-6)
all(abs(old - new_v2) < 1e-6)

fit_intercept = FALSE
old = sc.weight.fw(xy, eta, fit_intercept, lambda, min_decrease, max_iter)$lambda
new_v2 = sc.weight.fw.v2(xy, eta, fit_intercept, lambda, min_decrease, max_iter)$lambda
new = fw_sc_weight(xy, lambda, fit_intercept, eta, min_decrease, max_iter)$lambda
all(abs(old - new) < 1e-6)
all(abs(old - new_v2) < 1e-6)

contract3 = function(X, v) {
  stopifnot(length(dim(X)) == 3, dim(X)[3] == length(v))
  out = array(0, dim = dim(X)[1:2])
  if (length(v) == 0) { return(out) }
  for (ii in 1:length(v)) {
    out = out + v[ii] * X[, , ii]
  }
  return(out)
}
# A Frank-Wolfe + Gradient solver for lambda, omega, and beta when there are covariates
# Uses the exact line search Frank-Wolfe steps for lambda, omega and (1/t)*gradient steps for beta
# pass update.lambda=FALSE/update.omega=FALSE to fix those weights at initial values, defaulting to uniform 1/T0 and 1/N0
sc.weight.fw.covariates = function(Y, X = array(0, dim = c(dim(Y), 0)), zeta.lambda = 0, zeta.omega = 0,
                                   lambda.intercept = TRUE, omega.intercept = TRUE,
                                   min.decrease = 1e-3, max.iter = 1000,
                                   lambda = NULL, omega = NULL, beta = NULL, update.lambda = TRUE, update.omega = TRUE) {
  stopifnot(length(dim(Y)) == 2, length(dim(X)) == 3, all(dim(Y) == dim(X)[1:2]), all(is.finite(Y)), all(is.finite(X)))
  T0 = ncol(Y) - 1
  N0 = nrow(Y) - 1
  if (length(dim(X)) == 2) { dim(X) = c(dim(X), 1) }
  if (is.null(lambda)) {  lambda = rep(1 / T0, T0)   }
  if (is.null(omega)) {  omega = rep(1 / N0, N0)    }
  if (is.null(beta)) {  beta = rep(0, dim(X)[3]) }

  update.weights = function(Y, lambda, omega) {
    Y.lambda = if (lambda.intercept) { apply(Y[1:N0, ], 2, function(row) { row - mean(row) }) } else { Y[1:N0, ] }
    if (update.lambda) { lambda = fw.step(Y.lambda[, 1:T0], lambda, Y.lambda[, T0 + 1], N0 * Re(zeta.lambda^2)) }
    err.lambda = Y.lambda %*% c(lambda, -1)

    Y.omega = if (omega.intercept) { apply(t(Y[, 1:T0]), 2, function(row) { row - mean(row) }) } else { t(Y[, 1:T0]) }
    if (update.omega) { omega = fw.step(Y.omega[, 1:N0], omega, Y.omega[, N0 + 1], T0 * Re(zeta.omega^2)) }
    err.omega = Y.omega %*% c(omega, -1)

    val = Re(zeta.omega^2) * sum(omega^2) + Re(zeta.lambda^2) * sum(lambda^2) + sum(err.omega^2) / T0 + sum(err.lambda^2) / N0
    list(val = val, lambda = lambda, omega = omega, err.lambda = err.lambda, err.omega = err.omega)
  }

  vals = rep(NA, max.iter)
  t = 0
  print(glue::glue("t:{t},s:{0.0},b:[{v2s(beta)}],lambda:[{v2s(lambda)}],omega:[{v2s(omega)}]"))
  Y.beta = Y - contract3(X, beta)
  weights = update.weights(Y.beta, lambda, omega)
  # state is kept in weights$lambda, weights$omega, beta
  while (t < max.iter && (t < 2 || vals[t - 1] - vals[t] > min.decrease^2)) {
    t = t + 1
    grad.beta = -if (dim(X)[3] == 0) { c() } else {
      apply(X, 3, function(Xi) {
        t(weights$err.lambda) %*% Xi[1:N0, ] %*% c(weights$lambda, -1) / N0 +
          t(weights$err.omega) %*% t(Xi[, 1:T0]) %*% c(weights$omega, -1) / T0
      })
    }
    alpha = 1 / t
    beta = beta - alpha * grad.beta
    print(glue::glue("t:{t},s:{weights$val},b:[{v2s(beta)}],lambda:[{v2s(weights$lambda)}],omega:[{v2s(weights$omega)}]"))

    Y.beta = Y - contract3(X, beta)
    weights = update.weights(Y.beta, weights$lambda, weights$omega)
    vals[t] = weights$val
  }
  list(lambda = weights$lambda, omega = weights$omega, beta = beta, vals = vals)
}

set.seed(0)
m = 10
n = 6
v = 4
xty = matrix(rnorm(m * n), m, n)
cov = array(rnorm(m * n * v), dim = c(m, n, v))
beta = rep(0, v)

lambda = rep(1 / (n -  1), n - 1)
eta_lambda = 1e-6
fit_intercept_lambda = TRUE
update_lambda = TRUE

omega = rep(1 / (m - 1), m - 1)
eta_omega = 1e-6
fit_intercept_omega = TRUE
update_omega = TRUE

min_decrease = 1e-3
max_iter = 4

cov = array(0, dim = c(m, n, v))
old = sc.weight.fw.covariates(
  xty, cov,
  eta_lambda, eta_omega, fit_intercept_lambda, fit_intercept_omega,
  min_decrease, max_iter,
  lambda, omega, beta,
  update_lambda, update_omega
)

new = fw_sc_weight_covariates(
  xty, cov, beta,
  lambda, eta_lambda, fit_intercept_lambda, update_lambda,
  omega, eta_omega, fit_intercept_omega, update_omega,
  min_decrease, max_iter
)

c(
  all(abs(old$lambda - new$lambda) < 1e-6),
  all(abs(old$omega - new$omega) < 1e-6),
  all(abs(old$beta - new$beta) < 1e-6)
)


*/
