#include "ReddyEstimator.h"

cv::Mat ReddyEstimator::arma2cv(const arma::field<arma::vec>& matrix) {
    int ovlstep = N;

    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC(channels));

    for (int y =0, i=0;y < matrix.n_rows; y++, i+= ovlstep) {
        for (int x=0, j=0;x < matrix.n_cols; x++, j+= ovlstep) {
            arma::cube res_mtx(N, N, channels);
            int offset = N * N;

            for (int ch = 0; ch < channels; ch++)
            {
                res_mtx.slice(ch) = arma::reshape(matrix(y, x).rows(ch * offset, (ch + 1) * offset - 1), N, N);
            }


            for (int k = i, p = 0; k < (i + N); k++, p++)
            {
                for (int l = j, q = 0; l < (j + N); l++, q++)
                {
                    if (channels == 3)
                    {
                        cv::Vec3b pix;
                        for (int ch = 0; ch < channels; ch++)
                        {
                            pix[ch] = arma::u8(res_mtx.slice(ch)(p, q));
                        }
                        image.at<cv::Vec3b> (k, l) = pix;
                    }
                    else
                    {
                        image.at<arma::u8> (k, l) = arma::u8(res_mtx.slice(0)(p, q));
                    }
                }
            }
        }
    }

    return image;
}

arma::vec create_dct_table(int N)
{
    arma::vec DCT_coeffs;
    DCT_coeffs.set_size(N * N);

    int k = 0;
    double scale_fac_i;

    for (double m = 0; m < N; m++)
    {
        for (double n = 0; n < N; n++)
        {

            scale_fac_i = (m == 0) ? sqrt(1.0 / double(N)) : sqrt(2.0 / double(N));
            DCT_coeffs(k++) = scale_fac_i * std::cos(double((arma::math::pi() * m) / (2 * N) * (2 * n + 1)));
        }
    }

    return DCT_coeffs;
}

ReddyEstimator::ReddyEstimator(int N, int min_frames, double eta):
    N(N), min_frames(min_frames), eta(eta)
{
    arma::vec DCT_coeffs = create_dct_table(N * 2);

    dct_mtx = arma::ones<arma::mat> (N * 2, N* 2);

    //INITIALISE THE DCT MATRIX 8x8
    for (int i = 0; i < N * 2; i++)
    {
        for (int j = 0; j < N * 2; j++)
            dct_mtx(i, j) = DCT_coeffs[i * N * 2 + j];
    }

    //	Three neighbours offsets (default case)
    dst_offset.set_size(4, 6);
    src_offset.set_size(4, 6);
    curr_blk_offset.set_size(4, 2);

    //	One neighbour offsets
    src_offset_1N.set_size(4, 2);
    dst_offset_1N.set_size(4, 2);
    curr_blk_offset_1N.set_size(4, 2);

    //	 0  |  1
    //	____|_____
    //	 2  |  3
    //	    |

    //  0,1,2 & 3 are the possible positions of the unknown block. Each row in the
    //	table indicate the (x,y) locations of the destination buffer for 3 filled
    //	background neighbours in the clockwise direction.


    dst_offset
            = "\
	0 			 	16			16 				16  		16 			 	0;\
	0 		  	 	0 			16 				16  		16 			 	0;\
	0 		  	 	0 			0 			  	16  		16 			 	0;\
	0 		  	    0			0 			  	16  		16 				16";

    //	respective source indices of the filled neighbours in the same order.

    src_offset
            = " \
0		 1	    1		1 		1 		 0;\
0 		-1 		1 		0 	 	1 	    -1;\
-1 		-1 	   -1 		0 		0 		-1;\
-1 		 0 	   -1 		1 		0 		 1";

    //position of the unknown block, w.r.t to figure above,
    //which gets filled by candidates, one at a time.
    curr_blk_offset = " \
	 0		 0;\
	 0 		 16;\
	 16		 16;\
	 16 	 0";

    //Same table but with 1 neighbour at a time instead of 3
    dst_offset_1N = "\
	 0   0; \
	 0	 16; \
     16   0; \
     0   0";

    src_offset_1N = "\
	-1   0; \
	 0	 1; \
     1   0; \
     0  -1";

    curr_blk_offset_1N
            = " \
		 16		 0;\
		 0 		 0;\
		 0		 0;\
		 0 	 	 16";

    for (int i = 0; i < dst_offset.n_rows; ++i)
    {
        for (int j = 0; j < dst_offset.n_cols; ++j)
        {
            if (dst_offset(i, j) == 16)
                dst_offset(i, j) = N;
        }
    }

    for (int i = 0; i < curr_blk_offset.n_rows; ++i)
    {
        for (int j = 0; j < curr_blk_offset.n_cols; ++j)
        {
            if (curr_blk_offset(i, j) == 16)
                curr_blk_offset(i, j) = N;
        }
    }

    for (int i = 0; i < dst_offset_1N.n_rows; ++i)
    {
        for (int j = 0; j < dst_offset_1N.n_cols; ++j)
        {
            if (dst_offset_1N(i, j) == 16)
                dst_offset_1N(i, j) = N;
        }
    }

    for (int i = 0; i < curr_blk_offset_1N.n_rows; ++i)
    {
        for (int j = 0; j < curr_blk_offset_1N.n_cols; ++j)
        {
            if (curr_blk_offset_1N(i, j) == 16)
                curr_blk_offset_1N(i, j) = N;
        }
    }
}

int ReddyEstimator::find_atleast_3_filled_neighbours(int x, int y, const arma::imat& background_mask)
{
    int ovlstep = N;

    int xmb = (width - N) / ovlstep;
    int ymb = (height - N) / ovlstep;

    int intrpval = 0;
    int tr, lc, rc, br;
    int top, left, right, bottom;

    (y > 0) ? top = 1 : top = 0;
    (y < ymb) ? bottom = 1 : bottom = 0;

    (x > 0) ? left = 1 : left = 0;
    (x < xmb) ? right = 1 : right = 0;

    (top == 0) ? tr = 1 : tr = 0;
    (bottom == 0) ? br = 1 : br = 0;
    (right == 0) ? rc = 1 : rc = 0;
    (left == 0) ? lc = 1 : lc = 0;

    arma::imat mask_win1 = background_mask.submat((y - 1 + tr), (x - 1 + lc), (y + 1 - br), (x + 1 - rc));

    arma::imat win_mask = arma::ones<arma::imat> (3, 3);

    if (tr == 1 && br == 0 && rc == 0 && lc == 0)
    {
        win_mask.submat(1, 0, 2, 2) = mask_win1;
    }

    if (tr == 0 && br == 1 && rc == 0 && lc == 0)
    {
        win_mask.submat(0, 0, 1, 2) = mask_win1;
    }

    if (tr == 0 && br == 0 && rc == 1 && lc == 0)
    {
        win_mask.submat(0, 0, 2, 1) = mask_win1;
    }

    if (tr == 0 && br == 0 && rc == 0 && lc == 1)
    {
        win_mask.submat(0, 1, 2, 2) = mask_win1;
    }

    if (tr == 1 && br == 0 && rc == 0 && lc == 1)
    {
        win_mask.submat(1, 1, 2, 2) = mask_win1;
    }

    if (tr == 1 && br == 0 && rc == 1 && lc == 0)
    {
        win_mask.submat(1, 0, 2, 1) = mask_win1;
    }

    if (tr == 0 && br == 1 && rc == 0 && lc == 1)
    {
        win_mask.submat(0, 1, 1, 2) = mask_win1;
    }

    if (tr == 0 && br == 1 && rc == 1 && lc == 0)
    {
        win_mask.submat(0, 0, 1, 1) = mask_win1;
    }

    if (tr == 0 && br == 0 && rc == 0 && lc == 0)
    {
        win_mask = mask_win1;
    }

    int lval = 0, pow2 = 1;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            arma::imat tmp = win_mask.submat(i, j, (i + 1), (j + 1));

            if (arma::sum(arma::sum(tmp)) == 1)
                lval += pow2;
            pow2 *= 2;
        }
    }

    intrpval = lval;
    return intrpval;
}

int ReddyEstimator::unfilled_block_estimate_from_3_neighbours(int x, int y, int interpval,  const arma::field<std::vector<Cluster> >& clusters,const arma::field<arma::vec>& background, arma::mat &out_cost)
{
    int est_n = 0;
    arma::vec blk_arr(4);

    arma::mat res2(2 * N, 2 * N);

    switch (interpval)
    {
        case 1:
            //            %             LEFT-TOP ONLY
            blk_arr(0) = 2;
            est_n = 1;
            break;

        case 2:
            //            %             RIGHT-TOP ONLY
            blk_arr(0) = 3;
            est_n = 1;
            break;

        case 3:
            //            %             RIGHT-TOP & LEFT-TOP ONLY
            blk_arr(0) = 3;
            blk_arr(1) = 2;
            est_n = 2;
            break;

        case 4:
            //            %              LEFT-BOTTOM ONLY

            blk_arr(0) = 1;
            est_n = 1;
            break;

        case 5:
            //            %             LEFT-TOP & LEFT-BOTTOM ONLY
            blk_arr(0) = 1;
            blk_arr(1) = 2;
            est_n = 2;
            break;

        case 6:
            //            %           LEFT-BOTTOM & RIGHT-TOP ONLY
            blk_arr(0) = 2;
            blk_arr(1) = 3;
            est_n = 2;
            break;

        case 7:
            //            %             LEFT-TOP, LEFT-BOTTOM & RIGHT-TOP ONLY
            blk_arr(0) = 2;
            blk_arr(1) = 1;
            blk_arr(2) = 3;
            est_n = 3;
            break;

        case 8:
            //            %             RIGHT-BOTTOM ONLY
            blk_arr(0) = 0;
            est_n = 1;
            break;

        case 9:
            //            %				RIGHT-BOTTOM & LEFT-TOP,
            blk_arr(0) = 0;
            blk_arr(1) = 2;
            est_n = 2;
            break;
        case 10:
            //            %             RIGHT-TOP & RIGHT-BOTTOM ONLY
            blk_arr(0) = 3;
            blk_arr(1) = 0;
            est_n = 2;
            break;
        case 11:
            //            %  			RIGHT-TOP, LEFT-TOP & RIGHT-BOTTOM ONLY
            blk_arr(0) = 3;
            blk_arr(1) = 2;
            blk_arr(2) = 0;
            est_n = 3;
            break;
        case 12:
            //            %            RIGHT-BOTTOM & LEFT-BOTTOM ONLY
            blk_arr(0) = 0;
            blk_arr(1) = 1;
            est_n = 2;
            break;

        case 13:
            //						  RIGHT-BOTTOM, LEFT-TOP &  RIGHT-TOP
            blk_arr(0) = 0;
            blk_arr(1) = 2;
            blk_arr(2) = 3;
            est_n = 3;
            break;

        case 14:
            //						  RIGHT-BOTTOM, LEFT-BOTTOM &  RIGHT-TOP
            blk_arr(0) = 0;
            blk_arr(1) = 1;
            blk_arr(2) = 3;
            est_n = 3;
            break;

        case 15:
            //            %             ALL NEIGHBOURS
            blk_arr(0) = 0;
            blk_arr(1) = 1;
            blk_arr(2) = 2;
            blk_arr(3) = 3;
            est_n = 4;
            break;

        default:
            return -1;

    }

    arma::mat local_region = arma::zeros<arma::mat> (2 * N, 2 * N);

    out_cost.set_size(clusters(y, x).size(), est_n);

    //GET THE LIKELIHOODS
    arma::vec lhood = arma::zeros<arma::vec> (clusters(y, x).size());
    for (int i = 0; i < clusters(y, x).size(); ++i)
    {
        lhood(i) = clusters(y, x)[i].size;
        lhood(i) = std::min(min_frames, (int) lhood(i));
    }
    //		NORMALISE LIKLIHOODS
    lhood = lhood / arma::sum(lhood);

    int ofset = N * N;

    for (int k = 0; k < est_n; ++k)
    {
        arma::vec prior = arma::zeros<arma::vec> (clusters(y, x).size());
        for (int ch = 0; ch < channels; ++ch)
        {
            local_region.submat(dst_offset(blk_arr(k), 0), dst_offset(
                    blk_arr(k), 1), dst_offset(blk_arr(k), 0) + N - 1,
                    dst_offset(blk_arr(k), 1) + N - 1) = arma::reshape(
                        background(y + src_offset(blk_arr(k), 0),
                        x + src_offset(blk_arr(k), 1)).rows(ch * ofset, (ch + 1) * ofset - 1),
                        N,
                        N
                    );

            local_region.submat(dst_offset(blk_arr(k), 2), dst_offset(
                    blk_arr(k), 3), dst_offset(blk_arr(k), 2) + N - 1,
                    dst_offset(blk_arr(k), 3) + N - 1) = arma::reshape(
                        background(y + src_offset(blk_arr(k), 2),
                        x + src_offset(blk_arr(k), 3)).rows(ch * ofset, (ch+ 1) * ofset - 1),
                        N,
                        N
                    );

            local_region.submat(dst_offset(blk_arr(k), 4), dst_offset(
                    blk_arr(k), 5), dst_offset(blk_arr(k), 4) + N - 1,
                    dst_offset(blk_arr(k), 5) + N - 1) = arma::reshape(
                        background(y + src_offset(blk_arr(k), 4),
                        x+ src_offset(blk_arr(k), 5)).rows(ch * ofset, (ch + 1) * ofset - 1),
                        N,
                        N
                    );

            for (int i = 0; i < clusters(y, x).size(); ++i)
            {

                local_region.submat(
                        curr_blk_offset(blk_arr(k), 0),
                        curr_blk_offset(blk_arr(k), 1),
                        curr_blk_offset(blk_arr(k), 0) + N - 1,
                        curr_blk_offset(blk_arr(k), 1) + N - 1
                ) = arma::reshape(clusters(y, x)[i].representative.rows(ch * ofset, (ch + 1) * ofset - 1), N, N);

                res2 = dct_mtx * local_region * arma::trans(dct_mtx);
                res2(0, 0) = 0;
                prior(i) += arma::sum(arma::sum(arma::abs(res2)));

            }

        }

        //		NORMALISE PRIORS
        prior = prior / arma::sum(prior);

        for (int i = 0; i < clusters(y, x).size(); ++i)
        {
            prior(i) = exp(-prior(i) * clusters(y, x).size());
        }
        //		NORMALISE PRIORS AGAIN
        prior = prior / sum(prior);

        out_cost.col(k) = eta * log(prior) + log(lhood);
    }

    return 0;
}


int ReddyEstimator::find_atleast_1_filled_neighbour(int x, int y, const arma::imat& background_mask)
{
    int ovlstep = N;

    int xmb = (width - N) / ovlstep;
    int ymb = (height - N) / ovlstep;

    int intrpval = -1;
    int tr, lc, rc, br;
    int top, left, right, bottom;

    (y > 0) ? top = 1 : top = 0;
    (y < ymb - 1) ? bottom = 1 : bottom = 0;

    (x > 0) ? left = 1 : left = 0;
    (x < xmb - 1) ? right = 1 : right = 0;

    (top == 0) ? tr = 1 : tr = 0;
    (bottom == 0) ? br = 1 : br = 0;
    (right == 0) ? rc = 1 : rc = 0;
    (left == 0) ? lc = 1 : lc = 0;

    arma::imat mask_win1 = background_mask.submat((y - 1 + tr), (x - 1 + lc), (y + 1 - br), (x
            + 1 - rc));

    arma::imat win_mask = arma::ones<arma::imat> (3, 3);

    if (tr == 1 && br == 0 && rc == 0 && lc == 0)
    {
        win_mask.submat(1, 0, 2, 2) = mask_win1;

    }

    if (tr == 0 && br == 1 && rc == 0 && lc == 0)
    {
        win_mask.submat(0, 0, 1, 2) = mask_win1;

    }

    if (tr == 0 && br == 0 && rc == 1 && lc == 0)
    {
        win_mask.submat(0, 0, 2, 1) = mask_win1;

    }

    if (tr == 0 && br == 0 && rc == 0 && lc == 1)
    {
        win_mask.submat(0, 1, 2, 2) = mask_win1;

    }

    if (tr == 1 && br == 0 && rc == 0 && lc == 1)
    {
        win_mask.submat(1, 1, 2, 2) = mask_win1;

    }

    if (tr == 1 && br == 0 && rc == 1 && lc == 0)
    {
        win_mask.submat(1, 0, 2, 1) = mask_win1;

    }

    if (tr == 0 && br == 1 && rc == 0 && lc == 1)
    {
        win_mask.submat(0, 1, 1, 2) = mask_win1;

    }

    if (tr == 0 && br == 1 && rc == 1 && lc == 0)
    {
        win_mask.submat(0, 0, 1, 1) = mask_win1;

    }

    if (tr == 0 && br == 0 && rc == 0 && lc == 0)
    {
        win_mask = mask_win1;
    }

    int lval = 0, pow2 = 1;
    //CHECK TOP BLOCK
    if (win_mask(0, 1) == 0)
    {
        lval += pow2;
    }
    pow2 *= 2;
    //CHECK RIGHT BLOCK
    if (win_mask(1, 2) == 0)
    {
        lval += pow2;

    }
    pow2 *= 2;
    //CHECK BOTTOM BLOCK
    if (win_mask(2, 1) == 0)
    {
        lval += pow2;

    }
    pow2 *= 2;
    //CHECK LEFT BLOCK
    if (win_mask(1, 0) == 0)
    {
        lval += pow2;

    }

    switch (lval)
    {

        case 1:
            //		TOP
            intrpval = 0;
            break;
        case 2:
            //		RIGHT
            intrpval = 1;
            break;
        case 3:
            //		TOP,RIGHT
            intrpval = rand() % 2;
            break;
        case 4:
            //		BOTTOM
            intrpval = 2;
            break;
        case 5:
            //		TOP, BOTTOM
            intrpval = (rand() % 3) & 0x2;
            break;
        case 6:
            //		RIGHT,BOTTOM
            intrpval = (rand() % 2 + 1);
            break;
        case 7:
            //		TOP,RIGHT,BOTTOM
            intrpval = (rand() % 3);
            break;
        case 8:
            //		LEFT
            intrpval = 3;
            break;
        case 9:
            //		TOP,LEFT
            intrpval = 0;
            break;
        case 10:
            //		RIGHT,LEFT
            intrpval = 3;
            break;
        case 11:
            //		TOP,RIGHT,LEFT
            intrpval = 1;
            break;
        case 12:
            //		BOTTOM,LEFT
            intrpval = 2;
            break;
        case 13:
            //		TOP,BOTTOM,LEFT
            intrpval = 0;
            break;
        case 14:
            //		RIGHT,BOTTOM,LEFT
            intrpval = 3;
            break;
        case 15:
            //		ALL NEIGHBOURS
            intrpval = (rand() % 4);
            break;
        default:
            intrpval = -1;
    }

    return intrpval;
}

int ReddyEstimator::unfilled_block_estimate_from_1_neighbour(int x, int y, int interpval,  const arma::field<std::vector<Cluster> >& clusters,const arma::field<arma::vec>& background, arma::mat &out_cost)
{
    int blk_id, vertical_flg;

    arma::mat res2;
    arma::mat local_region;
    switch (interpval)
    {

        case 0:
            //            %             TOP ONLY
            blk_id = 0;
            break;

        case 1:
            //            %             RIGHT ONLY
            blk_id = 1;
            break;

        case 2:
            //            %           BOTTOM ONLY
            blk_id = 2;
            break;

        case 3:
            //            %             LEFT ONLY
            blk_id = 3;
            break;

        default:
            return -1;

    }

    if (blk_id == 0 || blk_id == 2)
    {
        local_region = arma::zeros<arma::mat> (2 * N, N);
        vertical_flg = 1;
        res2.set_size(2 * N, N);
    }
    else
    {
        local_region = arma::zeros<arma::mat> (N, 2 * N);
        vertical_flg = 0;
        res2.set_size(N, 2 * N);

    }
    out_cost.set_size(clusters(y, x).size(), 1);

    //GET THE LIKELIHOODS
    arma::vec lhood = arma::zeros<arma::vec> (clusters(y, x).size());
    for (int i = 0; i < clusters(y, x).size(); ++i)
    {
        lhood(i) = clusters(y, x)[i].size;
        lhood(i) = std::min(min_frames, (int) lhood(i));
    }
    //		NORMALISE LIKLIHOODS
    lhood = lhood / arma::sum(lhood);

    int offset = N * N;
    arma::vec prior = arma::zeros<arma::vec> (clusters(y, x).size());
    for (int ch = 0; ch < channels; ++ch)
    {
        local_region.submat(
                dst_offset_1N(blk_id, 0), dst_offset_1N(blk_id, 1),
                dst_offset_1N(blk_id, 0) + N - 1,
                dst_offset_1N(blk_id, 1) + N - 1
        ) = arma::reshape(background(y + src_offset_1N(blk_id, 0), x + src_offset_1N(blk_id, 1)).rows(ch * offset, (ch+ 1) * offset - 1), N, N);

        for (int i = 0; i < clusters(y, x).size(); ++i)
        {
            local_region.submat(
                    curr_blk_offset_1N(blk_id, 0),
                    curr_blk_offset_1N(blk_id, 1),
                    curr_blk_offset_1N(blk_id, 0) + N - 1,
                    curr_blk_offset_1N(blk_id, 1) + N - 1
            ) = arma::reshape(clusters(y, x)[i].representative.rows(ch * offset, (ch + 1) * offset - 1), N, N);

            if (vertical_flg)
            {
                res2 = dct_mtx * local_region;
                res2.row(0) = arma::zeros<arma::rowvec> (N);
            }
            else
            {
                res2 = local_region * trans(dct_mtx);
                res2.col(0) = arma::zeros<arma::vec> (N);
            }

            prior(i) += arma::sum(arma::sum(arma::abs(res2)));

        }

    }

    //	NORMALISE  PRIORS

    prior = prior / sum(prior);

    for (int i = 0; i < clusters(y, x).size(); ++i)
    {
        prior(i) = exp(-prior(i) * clusters(y, x).size());
    }
    //		NORMALISE PRIORS AGAIN
    prior = prior / arma::sum(prior);
    out_cost.col(0) = log(prior) + log(lhood);

    return 0;
}


void ReddyEstimator::mark_8_connected_neighbours(arma::imat &matrix)
{
    int ovlstep = N;

    int xmb = (width - N) / ovlstep;
    int ymb = (height - N) / ovlstep;

    int tr, lc, rc, br;
    int top, left, right, bottom;

    for (int y = 0; y < matrix.n_rows; y++)
    {
        for (int x = 0; x < matrix.n_cols; x++)
        {

            if (matrix(y, x) == -1)
            {

                (y > 0) ? top = 1 : top = 0;
                (y < ymb) ? bottom = 1 : bottom = 0;

                (x > 0) ? left = 1 : left = 0;
                (x < xmb) ? right = 1 : right = 0;

                (top == 0) ? tr = 1 : tr = 0;
                (bottom == 0) ? br = 1 : br = 0;
                (right == 0) ? rc = 1 : rc = 0;
                (left == 0) ? lc = 1 : lc = 0;

                //				READ THE 8-NEIGHBOURS INTO A MATRIX
                arma::imat mask_win = matrix.submat((y - 1 + tr), (x - 1 + lc), (y
                        + 1 - br), (x + 1 - rc));

                for (int j = 0; j < mask_win.n_rows; j++)
                {
                    for (int i = 0; i < mask_win.n_cols; i++)
                    {

                        if (mask_win(j, i) == 0)
                        {
                            mask_win(j, i) = 1;
                        }

                    }
                }
                //				WRITE BACK THE MATRIX AFTER SETTING THE 8-NEIGHBOURS, IF THEY ARE 0
                matrix.submat((y - 1 + tr), (x - 1 + lc), (y + 1 - br), (x + 1
                        - rc)) = mask_win;
                matrix(y, x) = 0;
            }
        }
    }

}

cv::Mat ReddyEstimator::estimate(const arma::field<std::vector<Cluster> >& clusters) {
    int ovlstep = N;

    int xmb = (width - N) / ovlstep;
    int ymb = (height - N) / ovlstep;

    arma::field<arma::vec> background_matrix;
    background_matrix.set_size((1 + ymb), (1 + xmb));

    for (int y = 0; y < (1 + ymb); ++y)
    {
        for (int x = 0; x < (1 + xmb); ++x)
        {
            background_matrix(y, x).set_size(N * N * channels);
            background_matrix(y, x).fill(0);
        }
    }

    arma::imat background_mask;
    background_mask.set_size((1 + ymb), (1 + xmb));
    background_mask.fill(1);

    arma::imat best_blk_id;
    best_blk_id.set_size((1 + ymb), (1 + xmb));
    best_blk_id.fill(-1);

    bool found = false;
    for (int y =0;y < clusters.n_rows; y++) {
        for (int x=0;x < clusters.n_cols; x++) {
            const std::vector<Cluster>& block = clusters(y, x);

            if (block.size() == 1) {
                found = true;
                const Cluster& seed_cluster = block[0];

                background_matrix(y, x) = seed_cluster.representative;
                background_mask(y, x) = 0;
                best_blk_id(y, x) = 0;
            }
        }
    }

    if (!found) {
        const Cluster& seed_cluster = clusters(0, 0)[0];
        background_matrix(0, 0) = seed_cluster.representative;
        background_mask(0,0) = 0;
        best_blk_id(0,0) = 0;
    }

    cv::imshow("background", arma2cv(background_matrix));
    cv::waitKey(0);

    while (arma::sum(arma::sum(background_mask)) != 0)
    {
        bool found = false;
        for (int y =0;y < clusters.n_rows; y++) {
            for (int x=0;x < clusters.n_cols; x++) {
                if (background_mask(y, x) == 1)
                {
                    int interpval = find_atleast_3_filled_neighbours(x, y, background_mask);
                    arma::mat cost(1, 1);
                    cost.set_size(1, 1);
                    cost(0, 0) = -1;

                    int best_blk_idx = unfilled_block_estimate_from_3_neighbours(x, y, interpval, clusters, background_matrix, cost);

                    if (best_blk_idx < 0)
                    {
                        continue;
                    }

                    arma::vec costvec = arma::sum(cost, 1);
                    arma::uvec best_id = arma::sort_index(costvec, 1);

                    background_matrix(y, x) = clusters(y, x)[best_id(0)].representative;
                    background_mask(y, x) = 0;
                    best_blk_id(y, x) = best_id(0);
                    found = true;

                    cv::imshow("background", arma2cv(background_matrix));
                    cv::waitKey(10);
                }
            }
        }

        if (!found) {
            found = false;
            for (int y =0;y < clusters.n_rows && !found; y++) {
                for (int x=0;x < clusters.n_cols; x++) {
                    if (background_mask(y, x) == 1)
                    {
                        int interpval = find_atleast_1_filled_neighbour(x, y, background_mask);
                        arma::mat cost(1, 1);
                        cost.set_size(1, 1);
                        cost(0, 0) = -1;

                        int best_blk_idx = unfilled_block_estimate_from_1_neighbour(x, y, interpval, clusters, background_matrix, cost);

                        if (best_blk_idx < 0)
                        {
                            continue;
                        }

                        arma::vec costvec = arma::sum(cost, 1);
                        arma::uvec best_id = arma::sort_index(costvec, 1);

                        background_matrix(y, x) = clusters(y, x)[best_id(0)].representative;
                        background_mask(y, x) = 0;
                        best_blk_id(y, x) = best_id(0);
                        found = true;

                        cv::imshow("background", arma2cv(background_matrix));
                        cv::waitKey(10);

                        break;
                    }
                }
            }
        }
    }

    arma::imat tmp;
    tmp.copy_size(background_mask);
    tmp.fill(0);

    for (int iter = 0; iter < iterations; ++iter)
    {
        do
        {
            for (int y =0;y < clusters.n_rows; y++) {
                for (int x=0;x < clusters.n_cols; x++) {
                    background_mask(y, x) = (iter == 0) ? 1 : background_mask(y, x);

                    if (background_mask(y, x) == 1)
                    {
                        int interpval = find_atleast_3_filled_neighbours(x, y, background_mask);

                        arma::mat cost(1, 1);
                        cost.set_size(1, 1);
                        cost(0, 0) = -1;
                        int best_blk_idx = unfilled_block_estimate_from_3_neighbours(x, y, interpval, clusters, background_matrix, cost);

                        if (best_blk_idx < 0)
                        {
                            interpval = find_atleast_1_filled_neighbour(x, y, background_mask);

                            cost.set_size(1, 1);
                            cost(0, 0) = -1;
                            unfilled_block_estimate_from_1_neighbour(x, y, interpval, clusters, background_matrix, cost);
                        }

                        arma::vec costvec = arma::sum(cost, 1);
                        arma::uvec best_id = arma::sort_index(costvec, 1);

                        background_matrix(y, x) = clusters(y, x)[best_id(0)].representative;
                        background_mask(y, x) = 0;

                        //						STORE THE CHANGES IN BEST BLOCK IDS
                        if (best_blk_id(y, x) != best_id(0))
                        {
                            tmp(y, x) = -1;
                            best_blk_id(y, x) = best_id(0);
                        }
                    }
                }
            }
        } while ((arma::sum(arma::sum(background_mask)) != 0));
        mark_8_connected_neighbours(tmp);
        background_mask = tmp;
        tmp.fill(0);
    }

    return arma2cv(background_matrix);
}