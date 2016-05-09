//
//
// FLIRTLib - Fast Laser Interesting Region Transform Library
// Copyright (C) 2009-2010 Gian Diego Tipaldi and Kai O. Arras
//
// This file is part of FLIRTLib.
//
// FLIRTLib is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FLIRTLib is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with FLIRTLib.  If not, see <http://www.gnu.org/licenses/>.
//

#include "RansacFeatureSetMatcher.h"

#include <boost/random.hpp>
#include <boost/random/uniform_smallint.hpp>
#include <sys/time.h>
#include <boost/foreach.hpp>
#include <flann/flann.hpp>

typedef std::pair<unsigned int, double> IndexedDistance;
bool indexedDistanceCompare(const IndexedDistance& first, const IndexedDistance& second)
{
  return first.second < second.second;
}


RansacFeatureSetMatcher::RansacFeatureSetMatcher(double acceptanceThreshold, double successProbability, double inlierProbability, double distanceThreshold, double rigidityThreshold, bool adaptive, bool inliersScore):
    AbstractFeatureSetMatcher(acceptanceThreshold),
    m_successProbability(successProbability),
    m_inlierProbability(inlierProbability),
    m_distanceThreshold(distanceThreshold),
    m_rigidityThreshold(rigidityThreshold),
    m_adaptive(adaptive),
    m_scoreInliersOnly(inliersScore),
    m_maxCorrespondences(std::numeric_limits<unsigned short>::max()),
    m_flann(false),
    m_flannChecks(64)
{}

double RansacFeatureSetMatcher::matchSets(const std::vector<InterestPoint *> &reference, const std::vector<InterestPoint *> &data, OrientedPoint2D& transformation) const
{
    std::vector< std::pair<InterestPoint *, InterestPoint *> > correspondences;
    return matchSets(reference, data, transformation, correspondences);
}

double RansacFeatureSetMatcher::matchSets(const std::vector<InterestPoint *> &reference, const std::vector<InterestPoint *> &data, OrientedPoint2D& transformation,
					  std::vector< std::pair<InterestPoint *, InterestPoint *> > &correspondences) const
{
    correspondences.clear();
    unsigned int iterations = m_adaptive ? 1e17 : ceil(log(1. - m_successProbability)/log(1. - m_inlierProbability * m_inlierProbability));
    
    // Compute possible correspondences based on 1-NN thresholding
    std::vector< std::pair<InterestPoint *, InterestPoint *> > possibleCorrespondences;
    std::vector<IndexedDistance> minDistances;

    if (!m_flann){
        for(unsigned int i = 0; i < data.size(); i++){
            double minCorrespondenceDistance = 1e17;
            unsigned int minCorrespondenceIndex = 0;
            for(unsigned int j = 0; j < reference.size(); j++){
                double distance = data[i]->getDescriptor()->distance(reference[j]->getDescriptor());
                if(distance < minCorrespondenceDistance){
                    minCorrespondenceDistance = distance;
                    minCorrespondenceIndex = j;
                }
             }

            if(minCorrespondenceDistance < m_distanceThreshold){
                possibleCorrespondences.push_back(std::make_pair(data[i], reference[minCorrespondenceIndex]));
                minDistances.push_back(std::make_pair(minDistances.size(), minCorrespondenceDistance));
            }
        }
    }
    else { /* m_flann */
        // Number of nearest neighbours to use
        const size_t nn = 1;
        const size_t data_size = data.size();
        const size_t reference_size = reference.size();

        if (!(data_size &&  reference_size)){
          return std::numeric_limits<double>::max();
        }

        // Create and fill the flann matrices from the data and reference descriptors
        std::vector<double> data_descriptors;
        std::vector<double> reference_descriptors;
        std::vector<int> indices(data_size*nn);
        std::vector<double> distances(data_size*nn);

        BOOST_FOREACH (const InterestPoint* ip, data){
            ip->getDescriptor()->getFlatDescription(data_descriptors);
        }

        BOOST_FOREACH (const InterestPoint* ip, reference){
            ip->getDescriptor()->getFlatDescription(reference_descriptors);
        }

        const size_t length = data_descriptors.size()/data_size;

        flann::Matrix<double> flann_query(&data_descriptors[0], data_size, length);
        flann::Matrix<double> flann_reference(&reference_descriptors[0], reference_size, length);
        flann::Matrix<int> flann_indices(&indices[0], flann_query.rows, nn);
        flann::Matrix<double> flann_distances(&distances[0], flann_query.rows, nn);

        // construct a randomized kd-tree index using 4 kd-trees
        flann::Index<flann::L2<double> > index(flann_reference, flann::KDTreeIndexParams(4));
        index.buildIndex();

        // do a knn search, using m_flannChecks checks
        index.knnSearch(flann_query, flann_indices, flann_distances, nn, flann::SearchParams(m_flannChecks));

        for (int i=0; i<data_size; ++i){
            possibleCorrespondences.push_back(std::make_pair(data[i], reference[indices[nn*i]]));
            minDistances.push_back(std::make_pair(minDistances.size(), distances[nn*i]));
        }
    }
    
    // Check if there are enough absolute matches 
    if(possibleCorrespondences.size() < 2){  
//      std::cout << "Not enough possible correspondences" << std::endl;
        return 1e17;
    }

    if (possibleCorrespondences.size() > m_maxCorrespondences){
        // sort the min distances
        std::sort(minDistances.begin(), minDistances.end(), indexedDistanceCompare);
        minDistances.resize(m_maxCorrespondences);
        std::vector< std::pair<InterestPoint *, InterestPoint *> > tempCorrespondences;
        tempCorrespondences.reserve(m_maxCorrespondences);

        BOOST_FOREACH (const IndexedDistance& id, minDistances) {
            tempCorrespondences.push_back(possibleCorrespondences[id.first]);
        }
        possibleCorrespondences = tempCorrespondences;
    }
    
    // Check if there are enough matches compared to the inlier probability 
    if(double(possibleCorrespondences.size()) * m_inlierProbability < 2){  
// 	std::cout << "Not enough possible correspondences for the inlier probability" << std::endl;
	return 1e17;
    }
    
    boost::mt19937 rng;
    boost::uniform_smallint<int> generator(0, possibleCorrespondences.size() - 1);
    
    // Main loop
    double minimumScore = 1e17;
    for(unsigned int i = 0; i < iterations; i++){
// 	std::cout << "\tIteration " << i << std::endl;
	unsigned int first = generator(rng);
	unsigned int second = generator(rng);
	while(second == first) second = generator(rng); // avoid useless samples
	std::pair< std::pair<InterestPoint *, InterestPoint *>, std::pair<InterestPoint *, InterestPoint *> > minimumSampleSet(possibleCorrespondences[first], possibleCorrespondences[second]);
	
	// Test rigidity
	const Point2D& diffFirst = possibleCorrespondences[first].first->getPosition() - possibleCorrespondences[second].first->getPosition();
	const Point2D& diffSecond = possibleCorrespondences[first].second->getPosition() - possibleCorrespondences[second].second->getPosition();
	double distanceFirst = diffFirst * diffFirst;
	double distanceSecond = diffSecond * diffSecond;
	if((distanceFirst - distanceSecond)*(distanceFirst - distanceSecond)/(8*(distanceFirst + distanceSecond	)) > m_rigidityThreshold){
// 	    std::cout << "\t\tRigidity failure" << std::endl;
	    continue;
	}
	
	// Compute hypothesis
	std::vector< std::pair<InterestPoint *, InterestPoint *> > inlierSet;
	OrientedPoint2D hypothesis = generateHypothesis(minimumSampleSet);
	
	// Verify hypothesis
	double score = verifyHypothesis(reference, data, hypothesis, inlierSet);
	if(score < minimumScore){
	    minimumScore = score;
	    transformation = hypothesis;
	    correspondences = inlierSet;
	    
	    // Adapt the number of iterations
	    if (m_adaptive){
		double inlierProbability = double(correspondences.size())/double(data.size());
		iterations = ceil(log(1. - m_successProbability)/log(1. - inlierProbability * inlierProbability));
	    }
	}
    }
    std::vector<std::pair<Point2D, Point2D> > pointCorrespondences(correspondences.size());
    for(unsigned int i = 0; i < correspondences.size(); i++){
	pointCorrespondences[i] = std::make_pair(correspondences[i].first->getPosition(), correspondences[i].second->getPosition());
    }
    compute2DPose(pointCorrespondences, transformation);
    double score = verifyHypothesis(reference, data, transformation, correspondences);
    if (m_scoreInliersOnly){
      // Modify the score to be the sum of the errors of the inliers only
      score -= (data.size()-correspondences.size())*m_acceptanceThreshold;
    }
    return score;   
}
