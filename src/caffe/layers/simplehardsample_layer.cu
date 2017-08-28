/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/simplehardsample_layer.hpp"

namespace caffe {


template <typename Dtype>
__global__ void back_diff(const int count, const Dtype * topdiff, Dtype * bottomdiff,Dtype thres) {
CUDA_KERNEL_LOOP(index, count) {
	Dtype v = topdiff[index];
	if(v >= thres || v <= -thres)
		bottomdiff[index]=v;
	else
		bottomdiff[index]=0;


}
}



template <typename Dtype>
void SimpleHardSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if(bottom.size()>2)
    {
        caffe_gpu_mul(bottom[0]->count(), bottom[0]->gpu_data(),bottom[2]->gpu_data(), top[0]->mutable_gpu_data());
    }
    else
    {
        caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
    }
}


template <typename Dtype>
void SimpleHardSampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      switch (this->layer_param_.simplehardsample_param().type())
      {
            case SimpleHardSampleParameter_Type_SORT:
                  this->Backward_gpu_sort(top,propagate_down,bottom);
                  break;
            case SimpleHardSampleParameter_Type_MEAN:
                  this->Backward_gpu_mean(top,propagate_down,bottom);
                  break;
            default:
                  LOG(FATAL) << "Unknown type method.";

      }


}

template <typename Dtype>
void SimpleHardSampleLayer<Dtype>::Backward_gpu_mean(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype * top_diff = top[0]->gpu_diff();
	Dtype * bottom_diff = bottom[0]->mutable_gpu_diff();
	const int count = top[0]->count();
	

	Dtype asum=0;
	Dtype * vp = &asum;
	caffe_gpu_asum(count, top_diff,vp);
	
	Dtype meandiff = 0;
	if(count > 0)
	{
		meandiff = asum / Dtype(count);
	}
	//LOG(INFO)<<"top_diff mean:"<<meandiff;

	back_diff<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,top_diff,bottom_diff,meandiff);
	CUDA_POST_KERNEL_CHECK;


	

}

template <typename Dtype>
void SimpleHardSampleLayer<Dtype>::Backward_gpu_sort(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
	

	if(std::fabs(remain_hard_rate_ -1.0) < FLT_EPS_)
	{
		caffe_copy(bottom[0]->count(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
		return;
	}

	const Dtype * topdiff = top[0]->cpu_diff();
	sorted_top_diff_.clear();
	Dtype v;
	for(int i=0;i<top[0]->count();i++)
	{
		v = std::fabs(topdiff[i]);
		if(v > FLT_EPS_)
			sorted_top_diff_.push_back(v);
	}
	if(sorted_top_diff_.size()<1)
		return;

	std::sort(sorted_top_diff_.begin(),sorted_top_diff_.end());
	
	int idx = (1-remain_hard_rate_) * float(sorted_top_diff_.size());
	Dtype thres = sorted_top_diff_[idx];
	
	//LOG(INFO)<<"hardrate:"<<remain_hard_rate_<<", thres:"<<thres<<", idx:"<<idx<<", sort diff size:"<<sorted_top_diff_.size();

	const Dtype * top_diff = top[0]->gpu_diff();
	Dtype * bottom_diff = bottom[0]->mutable_gpu_diff();
	const int count = bottom[0]->count();
	back_diff<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,top_diff,bottom_diff,thres);
	CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(SimpleHardSampleLayer);

}  // namespace caffe
