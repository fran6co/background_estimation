#include "ReddyClusterer.h"
#include "ssim.h"

inline double measure_L1_distance(const arma::vec &a, const arma::vec &b)
{
    return arma::sum(arma::sum(arma::abs(a - b)));
}

inline double measure_correlation_coefficient(const arma::vec &a, const arma::vec &b)
{
    double rr = 1;

    double rep_mu = arma::mean(a);
    double cur_mu = arma::mean(b);
    arma::vec sf_cur_vec = b - cur_mu;
    arma::vec sf_rep_vec = a - rep_mu;

    double nmr1 = sqrt(arma::sum(arma::sum(arma::square(sf_cur_vec))));
    double nmr2 = sqrt(arma::sum(arma::sum(arma::square(sf_rep_vec))));
    double nmr_res = nmr1 * nmr2;

    if (nmr_res != 0)
    {
        double ccv = arma::dot(sf_cur_vec, sf_rep_vec);
        rr = std::abs(ccv / nmr_res);
    }

    return rr;
}

bool ReddyClusterer::reddy_merge(const arma::vec &a, const arma::vec &b) {
    double distance = measure_L1_distance(a, b);

    if (distance < L1_threshold)
    {
        double correlation = measure_correlation_coefficient(a, b);

        if (correlation > corr_coef_threshold)
        {
            return true;
        }
    }

    return false;
}

inline cv::Mat arma2cv(const arma::vec& a, int N, int channels) {
    cv::Mat ca = cv::Mat::zeros(N, N, CV_8UC(channels));

    arma::cube res_mtx(N, N, channels);
    int offset = N * N;

    for (int ch = 0; ch < channels; ch++)
    {
        res_mtx.slice(ch) = arma::reshape(a.rows(ch * offset, (ch + 1) * offset - 1), N, N);
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cv::Vec3b pix;
            for (int ch = 0; ch < channels; ch++)
            {
                pix[ch] = arma::u8(res_mtx.slice(ch)(i, j));
            }
            ca.at<cv::Vec3b> (i, j) = pix;
        }
    }

    return ca;
}

bool ReddyClusterer::ssim_merge(const arma::vec &a, const arma::vec &b) {
    double index = calcSSIM(arma2cv(a, N, 3), arma2cv(b, N, 3));

    return index >= 0.95;
}

ReddyClusterer::ReddyClusterer()
{
    initialized = false;
}


void ReddyClusterer::initialize(int width, int height, int channels) {
    ovlstep = N;
    int xmb = (width - N) / ovlstep;
    int ymb = (height - N) / ovlstep;

    _clusters.set_size((1 + ymb), (1 + xmb));

    L1_threshold = (N * N * pixel_diff * channels);

    initialized = true;
}

void ReddyClusterer::process(const cv::Mat& frame) {
    int width = frame.cols, height = frame.rows, channels = frame.channels();

    if (!initialized) {
        initialize(width, height, channels);
    }

    arma::cube img_frame = arma::zeros<arma::cube> (height, width, channels);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (channels == 3)
            {
                cv::Vec3b pixel = frame.at<cv::Vec3b> (i, j);
                for (int ch = 0; ch < channels; ch++)
                {
                    img_frame.slice(ch)(i, j) = pixel[ch];
                }
            }
            else
            {
                img_frame.slice(0)(i, j) = frame.at<arma::u8> (i, j);
            }

        }
    }

    for (int i = 0, y = 0; i <= height - N; i += ovlstep, y++)
    {
        for (int j = 0, x = 0; j <= width - N; j += ovlstep, x++)
        {
            int offset = N * N;
            arma::vec patch(offset, 1);
            patch.set_size(offset * channels);
            for (int ch = 0; ch < channels; ch++)
            {
                arma::mat tmp = img_frame.slice(ch).submat(i, j, (i + N - 1), (j + N - 1));
                patch.rows(ch * offset, (ch + 1) * offset - 1) = arma::reshape(tmp, N * N, 1);
            }

            std::vector<Cluster>& clusters = _clusters(y, x);

            bool merged = false;
            for (std::vector<Cluster>::iterator it = clusters.begin();it != clusters.end(); it++)
            {
                if (ssim_merge(it->representative, patch)) {
                //if (reddy_merge(it->representative, patch)) {
                    merged = true;

                    it->representative = it->representative * it->size + patch;
                    it->size++;
                    it->representative /= it->size;
                    break;
                }
            }

            if (!merged)
            {
                Cluster cluster;
                cluster.representative = patch;
                cluster.size = 1;

                if (clusters.size() > 0 && clusters.back().size <= 2)
                {
                    clusters.pop_back();
                }

                clusters.push_back(cluster);
            }
        }
    }
}

const arma::field<std::vector<Cluster> >& ReddyClusterer::clusters() {
    return _clusters;
}