/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>
#include <unordered_map>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 1000;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_psi(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle particle = {i, dist_x(gen), dist_y(gen), dist_psi(gen), 1.0};
        particles.push_back(particle);
    }
    weights = vector<double>(num_particles);
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (Particle &particle : particles) {
        if (fabs(yaw_rate) > 0.0001) {
            particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
            particle.theta += yaw_rate * delta_t;
        } else {
            particle.x += velocity * delta_t * cos(particle.theta);
            particle.y += velocity * delta_t * sin(particle.theta);
        }

        particle.x += dist_x(gen);
        particle.y += dist_y(gen);
        particle.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    for (LandmarkObs &obs : observations) {
        double min_dist = numeric_limits<double>::max();

        for (LandmarkObs p : predicted) {
            double distance = dist(obs.x, obs.y, p.x, p.y);
            if (distance < min_dist) {
                min_dist = distance;
                obs.id = p.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double sigma_x_2 = pow(sigma_x, 2);
    double sigma_y_2 = pow(sigma_y, 2);
    double sigma_x_y = 2 * M_PI * sigma_x * sigma_y;

    for (Particle &particle : particles) {

        vector<LandmarkObs> predictions;
        unordered_map<int, LandmarkObs> pred_map;
        for (Map::single_landmark_s l : map_landmarks.landmark_list) {
            if (fabs(particle.x - l.x_f) <= sensor_range && fabs(particle.y - l.y_f) <= sensor_range) {
                LandmarkObs pred = {l.id_i, l.x_f, l.y_f};
                predictions.push_back(pred);
                pred_map[l.id_i] = pred;
            }
        }

        std::vector<LandmarkObs> transformed_observations;
        for (LandmarkObs obs : observations) {
            double t_x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
            double t_y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;
            transformed_observations.push_back(LandmarkObs{obs.id, t_x, t_y});
        }

        dataAssociation(predictions, transformed_observations);

        double weight = 1;
        vector<int> associations = vector<int>(transformed_observations.size());
        vector<double> sense_x = vector<double>(transformed_observations.size());
        vector<double> sense_y = vector<double>(transformed_observations.size());
        for (LandmarkObs obs : transformed_observations) {
            LandmarkObs pred = pred_map[obs.id];
            double p = exp(-pow(obs.x - pred.x, 2) / (2 * sigma_x_2) - pow(obs.y - pred.y, 2.0) /
                                                                       (2 * sigma_y_2)) /
                       sigma_x_y;
            weight *= p;
            associations.push_back(obs.id);
            sense_x.push_back(obs.x);
            sense_y.push_back(obs.y);
        }
        particle.weight = weight;
        SetAssociations(particle, associations, sense_x, sense_y);
    }
}


void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    for (int i = 0; i < particles.size(); i++) {
        weights[i] = particles[i].weight;
    }

    discrete_distribution<> d(weights.begin(), weights.end());

    vector<Particle> new_particles;
    for (int i = 0; i < particles.size(); i++) {
        new_particles.push_back(particles[d(gen)]);
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
