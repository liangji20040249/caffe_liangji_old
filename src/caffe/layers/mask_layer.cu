/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/layers/mask_layer.hpp"

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {


	caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
	return ;

	// DO NOT MASK IN FP 
    
	if(bottom[0]->channels() == bottom[1]->channels())
	{
		caffe_gpu_mul(bottom[0]->count(), bottom[0]->gpu_data(),bottom[1]->gpu_data(),top[0]->mutable_gpu_data());
	}
	else
	{
		Dtype * topdata = top[0]->mutable_gpu_data();
		const Dtype * maskdata = bottom[1]->gpu_data();
		const Dtype * bottomdata = bottom[0]->gpu_data();
		const int spcount = bottom[1]->count();
		for(int i=0;i<bottom[0]->channels();i++)
		{
			caffe_gpu_mul(spcount, bottomdata + i*spcount,maskdata,topdata + i*spcount);
		}
	}

}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    if (propagate_down[0]) {
        	
	if(bottom[0]->channels() == bottom[1]->channels())
	{
		caffe_gpu_mul(bottom[0]->count(), top[0]->gpu_diff(),bottom[1]->gpu_data(),bottom[0]->mutable_gpu_diff());
	}
	else
	{
		const Dtype * topdiff = top[0]->gpu_diff();
		const Dtype * maskdata = bottom[1]->gpu_data();
		Dtype * bottomdiff = bottom[0]->mutable_gpu_diff();
		const int spcount = bottom[1]->count();
		for(int i=0;i<bottom[0]->channels();i++)
		{
			caffe_gpu_mul(spcount, topdiff + i*spcount,maskdata,bottomdiff + i*spcount);
		}
	}
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskLayer);

}  // namespace caffe
