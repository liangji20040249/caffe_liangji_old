/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/mask_layer.hpp"

namespace caffe {
template <typename Dtype>
void MaskLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
   CHECK_EQ(bottom[0]->num(), bottom[1]->num())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->height(), bottom[1]->height())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->width(), bottom[1]->width())<< "Inputs must have the same dimension.";
   CHECK((bottom[0]->channels() == bottom[1]->channels()) ||  bottom[1]->channels()==1 )<< "mask channels should be 1 or the same as bottom 0.";
   
   top[0]->ReshapeLike(*bottom[0]);
  
}

template <typename Dtype>
void MaskLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
   //CHECK_EQ(bottom[0]->count(), bottom[1]->count())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->num(), bottom[1]->num())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->height(), bottom[1]->height())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->width(), bottom[1]->width())<< "Inputs must have the same dimension.";
   CHECK((bottom[0]->channels() == bottom[1]->channels()) ||  bottom[1]->channels()==1 )<< "mask channels should be 1 or the same as bottom 0.";
   
   top[0]->ReshapeLike(*bottom[0]);
   
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {


	caffe_copy(bottom[0]->count(),bottom[0]->cpu_data(),top[0]->mutable_cpu_data());
	return;
	//DO NOT MASK IN FP

	if(bottom[0]->channels() == bottom[1]->channels())
	{
		caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(),bottom[1]->cpu_data(),top[0]->mutable_cpu_data());
	}
	else
	{
		Dtype * topdata = top[0]->mutable_cpu_data();
		const Dtype * maskdata = bottom[1]->cpu_data();
		const Dtype * bottomdata = bottom[0]->cpu_data();
		const int spcount = bottom[1]->count();
		for(int i=0;i<bottom[0]->channels();i++)
		{
			caffe_mul(spcount, bottomdata + i*spcount,maskdata,topdata + i*spcount);
		}
	}
    
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
	
	if(bottom[0]->channels() == bottom[1]->channels())
	{
		caffe_mul(bottom[0]->count(), top[0]->cpu_diff(),bottom[1]->cpu_data(),bottom[0]->mutable_cpu_diff());
	}
	else
	{
		const Dtype * topdiff = top[0]->cpu_diff();
		const Dtype * maskdata = bottom[1]->cpu_data();
		Dtype * bottomdiff = bottom[0]->mutable_cpu_diff();
		const int spcount = bottom[1]->count();
		for(int i=0;i<bottom[0]->channels();i++)
		{
			caffe_mul(spcount, topdiff + i*spcount,maskdata,bottomdiff + i*spcount);
		}
	}

        
    }
}

#ifdef CPU_ONLY
STUB_GPU(MaskLayer);
#endif

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);

}  // namespace caffe
