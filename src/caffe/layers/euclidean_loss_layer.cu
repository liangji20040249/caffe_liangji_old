#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {


  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());

  if(bottom.size()==3)
  {
        CHECK(bottom[2]->height()==bottom[0]->height());
        CHECK(bottom[2]->width()==bottom[0]->width());
        CHECK(bottom[2]->channels()==1||bottom[2]->channels()==bottom[0]->channels());

        Dtype * diff_data = diff_.mutable_gpu_data();
        const Dtype * mask_data = bottom[2]->gpu_data();
        const int spdim = bottom[0]->height() * bottom[0]->width();
        const int bottom_data_count = bottom[0]->count();
        if(bottom[2]->channels()==1)
        {
                for(int i=0;i<bottom[0]->channels();i++)
                {
                 caffe_gpu_mul(spdim,diff_data + i*spdim,mask_data,diff_data + i*spdim);
                }
        }
        else
        {
                caffe_gpu_mul(bottom_data_count,diff_data ,mask_data,diff_data );
        }
  }


  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
