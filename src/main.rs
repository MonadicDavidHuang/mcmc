extern crate nalgebra as na;
extern crate gnuplot;
extern crate rand;

use na::*;
use std::f64;
use std::vec::Vec;
use rand::Rng;
use std::collections::BTreeMap;
use gnuplot::{Figure, Color};

const PI: f64 = std::f64::consts::PI;

// gaussian ditribution, with controable mean and standard deviation
fn gaussian_dist(x: & f64, mean: & f64, std_d: & f64) -> f64 {
    let in_exp: f64 = (-1.0) * (x - mean).powi(2) / (2.0 * std_d);
    let density: f64 = (1.0_f64.exp()).powf(in_exp) / (2.0 * PI * std_d).sqrt();
    density
}

// multi gaussian ditribution, with controable mean and standard deviation
fn multi_gaussian_dist(x_v: & DVector<f64>, mean_v: & DVector<f64>, std_d_mat: & DMatrix<f64>) -> f64 {
    let n_dim: usize = mean_v.nrows();

    let x_minus_mean = x_v - mean_v;

    let mut std_d_mat_cp = DMatrix::from_diagonal_element(n_dim, n_dim, 1.0);
    std_d_mat_cp.copy_from(std_d_mat);

    let std_d_mat_inv = match std_d_mat_cp.try_inverse() {
        Some(mat_inv) => mat_inv,
        None => panic!("Not regular matrix!"),
    };

    let mut in_exp = 0.0;
    let quad_form = x_minus_mean.transpose() * std_d_mat_inv * x_minus_mean;
    for &i in quad_form.iter() {in_exp = (-0.5) * i; break;}

    let det: f64 = std_d_mat.determinant();

    let base :f64 = (2.0 * PI).sqrt().powi(n_dim as i32) * det.sqrt();

    let density: f64 = (1.0_f64.exp()).powf(in_exp) / base;

    density
}

// generate num_var-th set of sample where each set's sample ~ N(0, 1) and its size is num_sample
fn box_muller(num_var: i32, num_sample: i32, mean: f64, std_d: f64) -> Vec<Vec<f64>> {
    let mut vec_of_vec: Vec<Vec<f64>> = vec![Vec::new(); 0]; // init vec_of_vec to 0-size vector

    let loop_num = f64::ceil((num_var as f64) / 2.0) as i32;

    for i in 0..loop_num {
        vec_of_vec.push(Vec::new()); vec_of_vec.push(Vec::new());

        let mut count = 0;
        loop {
            let z1: f64 = 2.0 * rand::thread_rng().gen::<f64>() - 1.0; // z1 ~ U(-1, 1)
            let z2: f64 = 2.0 * rand::thread_rng().gen::<f64>() - 1.0; // z2 ~ U(-1, 1)

            let r_sq = z1.powi(2) + z2.powi(2);

            if !(r_sq <= 1.0) {continue;}

            let y1: f64 = z1 * ((-2.0 * r_sq.ln()) / r_sq).powf(0.5);
            let y2: f64 = z2 * ((-2.0 * r_sq.ln()) / r_sq).powf(0.5);

            vec_of_vec[i as usize].push(y1 * std_d + mean);
            vec_of_vec[(i + 1) as usize].push(y2 * std_d + mean);

            count += 2;

            if count >= num_sample {break;}
        }
    }
    // if num_sample is odd, pop last sample since box-muller generate pair of independent saple's
    if vec_of_vec.len() > (num_var as usize) {vec_of_vec.pop();}

    vec_of_vec
}

fn multi_dim_normal_dist_sample(mean_v: & DVector<f64>, std_d_mat: & DMatrix<f64>) -> DVector<f64> {
    let n_dim: usize = mean_v.nrows();

    let mut std_d_mat_cp = DMatrix::from_diagonal_element(n_dim, n_dim, 1.0);
    std_d_mat_cp.copy_from(std_d_mat);

    let tmp = box_muller(n_dim as i32, 1, 0.0, 1.0);

    let mut svec: Vec<f64> = Vec::new();
    for i in & tmp {for j in i {svec.push(*j);}}

    let z: DVector<f64> = DVector::from_iterator(n_dim, svec.iter().cloned());

    let l_mat: DMatrix<f64> = match std_d_mat_cp.cholesky() {
        Some(mat) => mat.l(),
        None => panic!("Not positive defined matrix!"),
    };

    let multi_gaussian_dist_sample = l_mat * z + mean_v;

    multi_gaussian_dist_sample
}

// Metro-Polis Hasting
fn mph_sampling(num_sample: i32, n_dim: usize, target_dist: & Fn(& DVector<f64>) -> f64, proposed_dist: & Fn(& DVector<f64>, & DVector<f64>) -> f64, proposed_sampler: & Fn(& DVector<f64>) -> DVector<f64>) -> Vec<DVector<f64>> {
    let mut accepted_samples: Vec<DVector<f64>> = Vec::new();

    let mut new_point: DVector<f64> = DVector::new_random(n_dim); // initial sample
    let mut old_point: DVector<f64> = DVector::new_random(n_dim); // last sample

    old_point.copy_from(& new_point); // no owenership moved

    let mut acc_prob;
    let acc_prob_calc = |n: & DVector<f64>, c: & DVector<f64>| -> f64 {
        let u = 1e20 * target_dist(n) * proposed_dist(c, n);
        let l = 1e20 * target_dist(c) * proposed_dist(n, c);
        let last = if u == 0.0 && l == 0.0 {0.0} else if l == 0.0 {1.0} else {u / l};
        (1.0).min(last)
    };

    let mut count_burn = 0;
    loop {
        new_point = proposed_sampler(& old_point); // new ownership moved to new_point

        acc_prob = acc_prob_calc(& new_point, & old_point);
        let secret_number = rand::thread_rng().gen_range(0.0, 1.0);

        if secret_number <= acc_prob {
            count_burn += 1;
            let mut stock: DVector<f64> = DVector::new_random(n_dim); // new ownership moved to stock
            stock.copy_from(& new_point); // no owenership moved

            if count_burn > 5000 {
                accepted_samples.push(stock); // exist owenership moved from stock to accepted_samples
            }

            old_point = new_point; // exist owenership moved from new_point to old_point
        }
        else {println!("sample:{}   secret_number:{}   acc_prob:{}", accepted_samples.len(), secret_number, acc_prob);}

        if accepted_samples.len() >= (num_sample as usize) {break;}
    }

    accepted_samples
}

// return max and min element of vector
fn max_and_min(sample: & Vec<f64>) -> (f64, f64) {
    let mut mx = f64::MIN_POSITIVE;
    let mut mn = f64::MAX;

    for i in sample { // sample is already referer!
        if *i > mx {mx = *i;}
        if *i < mn {mn = *i;}
    }

    (mx, mn)
}

// serch nearest discretized area
fn nearest_search(base: & Vec<f64>, target: f64) -> f64 {
    let mut l = 0;
    let mut r = (*base).len();

    while l < r {
        let mid = ((l + r) as i32) / 2;
        if (*base)[mid as usize] == target {return target;}
        else if (*base)[mid as usize] > target {
            r = mid as usize;
        }
        else {
            l = (mid + 1) as usize;
        }
    }

    let last_index = if l >= (*base).len() {((*base).len() - 1) as usize} else {l};
    return (*base)[last_index];
}

// discretize one samples
fn discretize(samples: & Vec<f64>) -> Vec<f64> {
    let (mx, mn) = max_and_min(samples);
    println!("max: {}, min: {}", mx, mn);

    let window: f64 = 1.0;

    let mut base: Vec<f64> = Vec::new();

    let mut cur_point: f64 = mn;
    for _ in (mn as i32)..((mx as i32) + 1) {
        cur_point += window;
        let point = cur_point - (window / 2.0);
        base.push(point);
    }
    base.sort_by(|a, b| {a.partial_cmp(b).unwrap()});

    let mut discretized_samples: Vec<f64> = Vec::new();

    for e in samples {
        discretized_samples.push(nearest_search(& base, *e));
    }

    discretized_samples
}

// make histgram
fn counting(discretized_samples: & Vec<f64>) -> BTreeMap<String, f64> {
    let mut hist: BTreeMap<String, f64> = BTreeMap::new();

    for e in discretized_samples {
        let quantity = hist.entry((*e).to_string()).or_insert(0.0);
        *quantity += 1.0;
    }

    for (_, value) in hist.iter_mut() {
        *value /= (*discretized_samples).len() as f64;
    }

    hist
}

fn main() {
    let n_dim = 2;
    let mean: f64 = 0.0;
    let std_d: f64 = 500.0;
    let sample_num: i32 = 200000;

    // Metro-Polis Hasting method sampling from multi-dim gaussian
    // parameter for target dist
    let mean_vector_target: Vec<f64> = (vec![0.0; n_dim]).iter().map(|_| -> f64 {mean}).collect();
    let mean_v_target: DVector<f64> = DVector::from_iterator(n_dim, mean_vector_target.iter().cloned());
    let std_d_mat_target: DMatrix<f64> = DMatrix::from_diagonal_element(n_dim, n_dim, std_d);
    let target_dist = |x_v: & DVector<f64>| -> f64 {multi_gaussian_dist(x_v, & mean_v_target, & std_d_mat_target)};

    // parameter for proposed dist
    let std_d_mat_proposed: DMatrix<f64> = DMatrix::from_diagonal_element(n_dim, n_dim, std_d * 0.1);
    let proposed_dist = |x_v: & DVector<f64>, mean_v_propose: & DVector<f64>| -> f64 {multi_gaussian_dist(x_v, mean_v_propose, & std_d_mat_proposed)};
    let proposed_sampler = |mean_v: & DVector<f64>| -> DVector<f64> {multi_dim_normal_dist_sample(mean_v, & std_d_mat_proposed)};

    let samples_v: Vec<DVector<f64>> = mph_sampling(sample_num, n_dim, & target_dist, & proposed_dist, & proposed_sampler);
    let mut samples: Vec<f64> = Vec::new();

    for i in &samples_v {samples.push((*i)[0]);}

    let discretized_samples: Vec<f64> = discretize(&samples);

    let discretized_pair_map: BTreeMap<String, f64> = counting(& discretized_samples);

    let mut dense_vec_sampled: Vec<f64> = Vec::new();
    let mut element_vec_sampled: Vec<f64> = Vec::new();

    for (key, value) in discretized_pair_map.iter() {
        dense_vec_sampled.push(*value);
        element_vec_sampled.push((*key).parse::<f64>().unwrap());
    }

    // N(0, 1) +++++++++++++++++++++++++++++++++++++++++++++++++
    let std_normal_dist = |x| -> f64 {gaussian_dist(& x, & mean, & std_d)};

    let mut dense_vec: Vec<f64> = Vec::new();
    let mut element_vec: Vec<f64> = Vec::new();

    for i in ((-4.0 * std_d.sqrt() + mean) as i32)..((4.0 * std_d.sqrt() + mean) as i32) {
        let x: f64 = i as f64;
        element_vec.push(x);
        dense_vec.push(std_normal_dist(x));
    }

    /*
    // multi_dim_normal_dist_sample +++++++++++++++++++++++++++++++++++++++++++++++++
    let mean_v: DVector<f64> = DVector::from_iterator(2, [0.,0.].iter().cloned());
    let std_d_mat: DMatrix<f64> = DMatrix::from_diagonal_element(2, 2, std_d);

    let mut samples: Vec<f64> = Vec::new();
    for _ in 0..sample_num {
        samples.push(multi_dim_normal_dist_sample(& mean_v, & std_d_mat)[0]);
    }

    let discretized_samples: Vec<f64> = discretize(&samples);

    let discretized_pair_map: BTreeMap<String, f64> = counting(& discretized_samples);

    let mut dense_vec_sampled: Vec<f64> = Vec::new();
    let mut element_vec_sampled: Vec<f64> = Vec::new();

    for (key, value) in discretized_pair_map.iter() {
        dense_vec_sampled.push(*value);
        element_vec_sampled.push((*key).parse::<f64>().unwrap());
    }
    */

    /*
    // box_muller sampling +++++++++++++++++++++++++++++++++++++++++++++++++
    let samples_vector: Vec<Vec<f64>> = box_muller(1, sample_num, 0.0, std_d);
    let discretized_samples: Vec<f64> = discretize(&(samples_vector[0]));

    let discretized_pair_map: BTreeMap<String, f64> = counting(& discretized_samples);

    let mut dense_vec_sampled: Vec<f64> = Vec::new();
    let mut element_vec_sampled: Vec<f64> = Vec::new();

    for (key, value) in discretized_pair_map.iter() {
        dense_vec_sampled.push(*value);
        element_vec_sampled.push((*key).parse::<f64>().unwrap());
    }
    */


    // graph drawing +++++++++++++++++++++++++++++++++++++++++++++++++
    let mut fg = Figure::new();

    fg.axes2d()
    .lines(&element_vec, &dense_vec, &[Color("blue")])
    .points(&element_vec_sampled, &dense_vec_sampled, &[Color("red")]);

    fg.set_terminal("png", "test.png");
    fg.show();

    //linear algebra +++++++++++++++++++++++++++++++++++++++++++++++++
    /*
    let mean_v: DVector<f64> = DVector::from_iterator(1, [0.,].iter().cloned());
    let std_d_mat: DMatrix<f64> = DMatrix::from_diagonal_element(1, 1, 1.0);

    let x_v: DVector<f64> = DVector::from_iterator(1, [0.,].iter().cloned());

    let dens = multi_gaussian_dist(&x_v, &mean_v, &std_d_mat);
    println!("{}:{}", dens, gaussian_dist(& 0.0, & 0.0, & 1.0));

    let b: DVector<f64> = DVector::from_iterator(3, [1.,2.,3.,].iter().cloned());
    println!("{}",b);

    let std_d_mat: DMatrix<f64> = DMatrix::from_iterator(3, 3, [
        2.,2.,2.,
        2.,2.,2.,
        2.,2.,2.,
        ].iter().cloned());
    */
}
