#ifndef __ReddyClusterer_H_
#define __ReddyClusterer_H_

#include <opencv2/opencv.hpp>
#include <armadillo>
#include <vector>

struct Cluster
{
    unsigned int size;
    arma::vec representative;
};

cv::Mat arma2cv(const arma::field<arma::vec>& matrix, int N, int channels);

class ReddyClusterer{
    public:

    ReddyClusterer();

    void process(const cv::Mat& frame);
    const arma::field<std::vector<Cluster> >& clusters();

    int N;
    int ovlstep;
    double pixel_diff;
    double corr_coef_threshold;
    double L1_threshold;

    private:
    void initialize(int width, int height, int channels);
    bool initialized;

    bool reddy_merge(const arma::vec &a, const arma::vec &b);
    bool ssim_merge(const arma::vec &a, const arma::vec &b);

    arma::field<std::vector<Cluster> > _clusters;

};


#endif //__ReddyClusterer_H_
