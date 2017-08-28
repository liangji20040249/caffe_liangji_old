/*
 * Author: Liangji
 * Email: liangji20040249@gmail.com
 */
#include <vector>
#include <cfloat>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/simplehardsample_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
	template <typename Dtype>
	void SimpleHardSampleLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*> & bottom, const vector<Blob<Dtype>*> & top )
	{
		
		CHECK(bottom.size()==1);
		CHECK(top.size()==1);
		

		top[0]->ReshapeLike( *bottom[0] );
		
		remain_hard_rate_	= this->layer_param_.simplehardsample_param().remain_hard_rate();
		FLT_EPS_ =	1e-6;
		CHECK(remain_hard_rate_>=0 && remain_hard_rate_ <=1);
		/* use_balancesample_ = this->layer_param_.hardsample_param().use_balancesample(); */
	}


	template <typename Dtype>
	void SimpleHardSampleLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*> & bottom, const vector<Blob<Dtype>*> & top )
	{
		;
	}


	template <typename Dtype>
	void SimpleHardSampleLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*> & bottom,
						  const vector<Blob<Dtype>*> & top )
	{
		if ( bottom.size() > 2 )
		{
			caffe_mul( bottom[0]->count(), bottom[0]->cpu_data(), bottom[2]->cpu_data(), top[0]->mutable_cpu_data() );
		}else  {
			caffe_copy( bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data() );
		}
	}


	template <typename Dtype>
	void SimpleHardSampleLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*> & top,
						   const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom )
	{
		this->Backward_gpu( top, propagate_down,  bottom );
	}


	



#ifdef CPU_ONLY
	STUB_GPU( SimpleHardSampleLayer );
#endif

	INSTANTIATE_CLASS( SimpleHardSampleLayer );
	REGISTER_LAYER_CLASS( SimpleHardSample );
}  /* namespace caffe */
