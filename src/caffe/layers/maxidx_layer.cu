#include <vector>

#include "caffe/layers/maxidx_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype get_data(Dtype * data, int num, int channels,int height, int width,int n,int c,int h,int w)
{
	if(h<0 || h >=height)
		return 0;
	if(w<0 || w >= width)
		return 0;
	
	return data[n*channels*height*width + c * height*width + h * width + w];
}

template <typename Dtype>
__device__ void set_data(Dtype * data, int num, int channels,int height, int width,int n,int c,int h,int w,Dtype v)
{
	if(h<0 || h >=height)
		return ;
	if(w<0 || w >= width)
		return ;
	

	data[n*channels*height*width + c * height*width + h * width + w]=v;
}




template <typename Dtype>
__global__ void forward_idx(const int count, const Dtype * bottom_data,Dtype * top_data,int num,int channels,int height,int width,int sub_channels,int idx_num)
{
CUDA_KERNEL_LOOP(index, count) {

	
	int n,c,h,w;
	int temp=index;

	n = temp / (height*width);
	temp = temp % (height*width);
	h = temp / width;
	temp = temp % width;
	w = temp;

	int idx=-1;
	Dtype maxvalue=0;
	Dtype curvalue=0;

	for(int i=0;i<idx_num;i++)
	{
		curvalue = 0;
		for(int j=0;j<sub_channels;j++)
		{
			Dtype v = get_data(bottom_data,  num,  channels, height,  width, n,i*sub_channels + j, h, w);
			curvalue = curvalue + v*v;
		}
		if(idx<0)
		{
			idx = 0;
			maxvalue = curvalue;
		}
		else
		{
			if(curvalue > maxvalue)
			{
				idx = i;
				maxvalue = curvalue;
			}
		}
	}
	set_data(top_data,  num,  idx_num, height,  width, n, idx, h, w,Dtype(1.0));
}

}

template <typename Dtype>
void MaxIdxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
	

	const int count = height_ * width_ * num_;
	const Dtype * bottom_data = bottom[0]->gpu_data();
	Dtype * top_data = top[0]->mutable_gpu_data();
	int sub_channels = channels_ / idx_num_;

	caffe_gpu_set(top[0]->count(),Dtype(0),top_data);
	
	
	forward_idx<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,bottom_data,top_data,num_,channels_,height_,width_,sub_channels,idx_num_);

	CUDA_POST_KERNEL_CHECK;
	
}

template <typename Dtype>
void MaxIdxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 ;
}

INSTANTIATE_LAYER_GPU_FUNCS(MaxIdxLayer);

}  // namespace caffe
