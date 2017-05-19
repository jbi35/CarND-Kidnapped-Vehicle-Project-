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

#include "particle_filter.h"

// random generator
std::random_device my_random_normal_gen;
// using mersenne twister engine for high quality samples
std::mt19937 std_normal_gen(my_random_normal_gen());
std::normal_distribution<double> std_normal_distribution(0,1);

double log_like_i (const double x,const double y, const double sig_x, const double sig_y)
{
  /*
  * calculate log likelihood based on bivariate normal distribution
  */
  return (-(x*x/(2*sig_x*sig_x) + y*y/(2*sig_y*sig_y))) - log((2.0*M_PI*sig_x*sig_y));
}

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /*
  * init initializes particles based on an itial noisy estimate
  */
	num_particles = 100;
	weights.reserve(num_particles);

	for (int i = 0; i < num_particles; i++)
	{
		Particle my_temp_particle;
		// init fields of the particle
		my_temp_particle.id 			= i;
		my_temp_particle.x 			  = x + std[0]*std_normal_distribution(std_normal_gen);
		my_temp_particle.y 			  = y + std[1]*std_normal_distribution(std_normal_gen);
		my_temp_particle.theta 	  = theta + std[2]*std_normal_distribution(std_normal_gen);
		my_temp_particle.weight 	= 1.0;
		weights[i] 					      = 1.0;
		particles.push_back(my_temp_particle);
	}
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m],
   *  standard deviation of y [m] standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */

   for (int i = 0; i < num_particles; i++)
   {

     // prevend division through zero as in lectures
   	if (fabs(yaw_rate)>0.0001)
    {
  		particles[i].x += velocity/yaw_rate*(sin(particles[i].theta
                        + yaw_rate*delta_t) - sin(particles[i].theta))
                        + std_pos[0]*std_normal_distribution(std_normal_gen);

  		particles[i].y +=  -velocity/yaw_rate*(cos(particles[i].theta
                         + yaw_rate*delta_t) - cos(particles[i].theta))
                         + std_pos[1]*std_normal_distribution(std_normal_gen);

  		particles[i].theta += yaw_rate*delta_t+ std_pos[2]
                            * std_normal_distribution(std_normal_gen);
  	}
    else
    {
			particles[i].x += velocity * cos(particles[i].theta) * delta_t
                        + std_pos[0]*std_normal_distribution(std_normal_gen);
			particles[i].y += velocity*sin(particles[i].theta)*delta_t
                        + std_pos[1]*std_normal_distribution(std_normal_gen);

			particles[i].theta += yaw_rate*delta_t + std_pos[2]
                            * std_normal_distribution(std_normal_gen);
		}

	} // eof loop particels
}
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> map_landmarks,
                                     std::vector<LandmarkObs>& observations_landmarks)
{
  /**
   * dataAssociation Solves the data association problem using a simple
   * nearest neighbor search
   *
   * @param map_landmarks vector containing map landmarks
   * @param observations_landmarks vector containing observed landmarks
   */

	double current_distance;
	// loop over all observations
	for (int i=0;i<observations_landmarks.size();i++)
	{
		double min_distance = 1000000;
		int min_distance_id = -1;
    // loop over all predictions
		for (int j=0;j<map_landmarks.size();j++)
		{
      // compute distance using helper function
			current_distance = dist(observations_landmarks[i].x,
                              observations_landmarks[i].y,
                              map_landmarks[j].x, map_landmarks[j].y);

			if (current_distance<min_distance)
			{
				min_distance = current_distance;
				// map landmarks are possibly filtered so don't use original id
        min_distance_id = j;
			}
		}
		observations_landmarks[i].id  = min_distance_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		                               std::vector<LandmarkObs> observations,
                                   Map map_landmarks)
{
  /**
   * updateWeights update weights of the particles based on sensor measurement
   *
   * @param sensor_range range of sensor
   * @param std_landmark  std deviation of measurement noise
   * @param observations sensor measurements (landmarks based)
   * @param map_landmarks map based on landmarks
   */
	weights.clear();

	for (int i_p=0;i_p<particles.size();i_p++)
	{
  	// convert observations to map coordinate system
		std::vector<LandmarkObs> observations_map_cosy;
    observations_map_cosy.reserve(observations.size());

    for (int i=0;i<observations.size();i++)
		{
      LandmarkObs temp_observation;
      // coordinate transform

			temp_observation.x = particles[i_p].x
                           + observations[i].x * cos(particles[i_p].theta)
                           - observations[i].y * sin(particles[i_p].theta);

      temp_observation.y = particles[i_p].y
                           + observations[i].x * sin(particles[i_p].theta)
                           + observations[i].y * cos(particles[i_p].theta);

      // assing temporary to be filled later with associated closest map landmark ID
			temp_observation.id = -1;

      observations_map_cosy.push_back(temp_observation);
		}

    // before solving the data association problem filter out landmarks
    // beyond sensor range
    std::vector<LandmarkObs> filtered_landmarks;
		for (int i=0;i<map_landmarks.landmark_list.size();i++)
		{
      // filter by  approximate distance to safe time
      if (std::abs(particles[i_p].x-map_landmarks.landmark_list[i].x_f)<=sensor_range &&
          std::abs(particles[i_p].y-map_landmarks.landmark_list[i].y_f)<=sensor_range)
			{
				LandmarkObs my_land_mark;
				my_land_mark.id = map_landmarks.landmark_list[i].id_i;
				my_land_mark.x = map_landmarks.landmark_list[i].x_f;
				my_land_mark.y = map_landmarks.landmark_list[i].y_f;

			 	filtered_landmarks.push_back(my_land_mark);
		 	}
		}

    // solve data association problem
		dataAssociation(filtered_landmarks, observations_map_cosy);

    // compute likelihood
    double log_like = 0.0;
    for (int j =0; j<observations_map_cosy.size(); j++)
    {
      const double x_dist = observations_map_cosy[j].x
                            - filtered_landmarks[observations_map_cosy[j].id].x;

      const double y_dist = observations_map_cosy[j].y
                            - filtered_landmarks[observations_map_cosy[j].id].y;

      log_like += log_like_i(x_dist,y_dist,std_landmark[0], std_landmark[1]);
    }
    weights.push_back(exp(log_like));
    particles[i_p].weight = exp(log_like);
	}
}

void ParticleFilter::resample()
{
  /**
   * resample  performs multinomial resampling
   */

  // random generator
	std::random_device my_random_gen;
	std::mt19937 generator(my_random_gen());

	// setup discrete distribution for weights
	std::discrete_distribution<int> my_weight_distribution(weights.begin(),
                                                         weights.end());

	std::vector<Particle> new_particles;
  // reserve some space
  new_particles.reserve(num_particles);


	for(int i=0;i<num_particles;i++)
	{
      Particle my_temp_particle;
			my_temp_particle = particles[my_weight_distribution(generator)];
	    new_particles.push_back(my_temp_particle);
	}
  // assign to class variable
	particles = new_particles;
}

void ParticleFilter::write(std::string filename)
{
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i)
  {
		dataFile << particles[i].x << " " << particles[i].y
             << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
