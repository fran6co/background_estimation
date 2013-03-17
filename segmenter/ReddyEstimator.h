#ifndef __ReddyEstimator_H_
#define __ReddyEstimator_H_

#include "ReddyClusterer.h"

class ReddyEstimator {
    public:
    ReddyEstimator(int N, int min_frames, double eta);

    cv::Mat estimate(const arma::field<std::vector<Cluster> >& clusters);
    int width;
    int height;
    int channels;
    int N;
    int min_frames;
    int iterations;
    double eta;

    private:

    arma::imat dst_offset, src_offset, curr_blk_offset;
    arma::imat dst_offset_1N, src_offset_1N, curr_blk_offset_1N;
    arma::mat dct_mtx;

    cv::Mat arma2cv(const arma::field<arma::vec>& matrix);

    void mark_8_connected_neighbours(arma::imat &matrix);
    int find_atleast_3_filled_neighbours(int x, int y, const arma::imat& background_mask);
    int unfilled_block_estimate_from_3_neighbours(int x, int y, int interpval,  const arma::field<std::vector<Cluster> >& clusters, const arma::field<arma::vec>& background, arma::mat &out_cost);
    int find_atleast_1_filled_neighbour(int x, int y, const arma::imat& background_mask);
    int unfilled_block_estimate_from_1_neighbour(int x, int y, int interpval,  const arma::field<std::vector<Cluster> >& clusters, const arma::field<arma::vec>& background, arma::mat &out_cost);
};


#endif //__ReddyEstimator_H_
