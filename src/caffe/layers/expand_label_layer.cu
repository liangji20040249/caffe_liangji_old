/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/expand_label_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void cal_select_one_label(const int count, Dtype select_one_label,int num,int outchannels,int height,int width,const Dtype * bottom_data,Dtype * top_data) 
{
      CUDA_KERNEL_LOOP(index, count) {
            int v = index;
	    int topdim = height*width*2;
            int spdim = height * width;
            int n,c,h,w;
            c=0;

            n = v / spdim;
            v = v % spdim;
            h = v / width;
            v = v % width;
            w = v;

            Dtype value = bottom_data[n*spdim + 0 + h*width + w];
	    if(int(value) != int(select_one_label))
            {
                  top_data[n*topdim + int(0)*spdim + h*width + w] = 1;
            }
            else
            {
                  top_data[n*topdim + int(1)*spdim + h*width + w] = 1;
            }
            
      }
}

template <typename Dtype>
__global__ void cal_select_one_label_withignore(const int count, Dtype select_one_label,int num,int outchannels,int height,int width,const Dtype * bottom_data,Dtype * top_data,Dtype * ignore_data, Dtype ignore_label) 
{
      CUDA_KERNEL_LOOP(index, count) {
            int v = index;
	    int topdim = height * width *2;
            int spdim = height * width;
            int n,c,h,w;
            c=0;

            n = v / spdim;
            v = v % spdim;
            h = v / width;
            v = v % width;
            w = v;

            Dtype value = bottom_data[n*spdim + 0 + h*width + w];

            if(int(value) == int(ignore_label))
            {
                  ignore_data[n*spdim + 0 + h*width + w] = 0;
                  top_data[n*topdim + int(0)*spdim + h*width + w] = 1;
            }
            else
            {
            if(int(value) != int(select_one_label))
            {
                  top_data[n*topdim + int(0)*spdim + h*width + w] = 1;
            }
            else
            {
                  top_data[n*topdim + int(1)*spdim + h*width + w] = 1;
            }
            }


      }
}


template <typename Dtype>
__global__ void cal_expand_label(const int count, int num,int outchannels,int height,int width,const Dtype * bottom_data,Dtype * top_data) 
{
      CUDA_KERNEL_LOOP(index, count) {
            int v = index;
            int spdim = height * width;
            int topdim = height * width * outchannels;
            int n,c,h,w;
            c=0;

            n = v / spdim;
            v = v % spdim;
            h = v / width;
            v = v % width;
            w = v;

            Dtype value = bottom_data[n*spdim + 0 + h*width + w];
	    if(value < 0 || value >= outchannels)
		return;
            top_data[n*topdim + int(value)*spdim + h*width + w] = 1;
      }
}

template <typename Dtype>
__global__ void cal_expand_label_withignore(const int count, int num,int outchannels,int height,int width,const Dtype * bottom_data,Dtype * top_data,Dtype * ignore_data, Dtype ignore_label) 
{
      CUDA_KERNEL_LOOP(index, count) {
            int v = index;
            int spdim = height * width;
	    int topdim = height * width * outchannels;
            int n,c,h,w;
            c=0;

            n = v / spdim;
            v = v % spdim;
            h = v / width;
            v = v % width;
            w = v;

            Dtype value = bottom_data[n*spdim + 0 + h*width + w];

	    if(int(value) == int(ignore_label))
	    {
		ignore_data[n*spdim + 0 + h*width + w] = 0;
	    }
	    else
            {
	      if(value < 0 || value >= outchannels)
                  ;
              else
                  top_data[n*topdim + int(value)*spdim + h*width + w] = 1;
            }
      }
}


template <typename Dtype>
void ExpandLabelLayer<Dtype>::Forward_expand_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

      int num = bottom[0]->num();
      int channels = bottom[0]->channels();
      int height = bottom[0]->height();
      int width = bottom[0]->width();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      
      int outchannels = top[0]->channels();

      caffe_gpu_set(top[0]->count(),Dtype(0),top_data);


      const int count = bottom[0]->num() * bottom[0]->height() * bottom[0]->width();

      if(top.size() == 1)
      {
            cal_expand_label<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, num,outchannels,height,width,bottom_data,top_data);
            CUDA_POST_KERNEL_CHECK;
      }
      else
      {
            Dtype* ignore_data = top[1]->mutable_gpu_data();
            caffe_gpu_set(top[1]->count(),Dtype(1),ignore_data);

            cal_expand_label_withignore<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, num,outchannels,height,width,bottom_data,top_data,ignore_data,ignore_label_);
            CUDA_POST_KERNEL_CHECK;
      }

}


template <typename Dtype>
void ExpandLabelLayer<Dtype>::Forward_select_one_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

      int num = bottom[0]->num();
      int channels = bottom[0]->channels();
      int height = bottom[0]->height();
      int width = bottom[0]->width();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      
      int outchannels = top[0]->channels();

      caffe_gpu_set(top[0]->count(),Dtype(0),top_data);

      //int sp_dim = bottom[0]->count() / bottom[0]->num(); 
      int sp_dim = bottom[0]->height() * bottom[0]->width();

      Dtype select_one_label = this->layer_param_.expand_label_param().select_one_label();

      const int count = bottom[0]->num() * sp_dim;
      if(top.size() == 1)
      {
            cal_select_one_label<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count, select_one_label,num,outchannels,height,width,bottom_data,top_data);
            CUDA_POST_KERNEL_CHECK;
      }
      else
      {
            Dtype* ignore_data = top[1]->mutable_gpu_data();
            caffe_gpu_set(top[1]->count(),Dtype(1),ignore_data);

            cal_select_one_label_withignore<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count, select_one_label,num,outchannels,height,width,bottom_data,top_data,ignore_data,ignore_label_);
            CUDA_POST_KERNEL_CHECK;
      }

}

template <typename Dtype>
void ExpandLabelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

      int num = bottom[0]->num();
      int channels = bottom[0]->channels();
      int height = bottom[0]->height();
      int width = bottom[0]->width();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      Dtype* top_data = top[0]->mutable_gpu_data();
      
      int outchannels = top[0]->channels();

      caffe_gpu_set(top[0]->count(),Dtype(0),top_data);

      int sp_dim = bottom[0]->count() / bottom[0]->num(); 

      

      switch (this->layer_param_.expand_label_param().type())
      {
            case ExpandLabelParameter_Type_EXPAND:
                  this->Forward_expand_gpu(bottom,top);
                  break;
            case ExpandLabelParameter_Type_SELECT_ONE:
                  this->Forward_select_one_gpu(bottom,top);
                  break;
            default:
                  LOG(FATAL) << "Unknown type method.";

      }

}

/// @brief refer to CPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void ExpandLabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		CHECK(false);
  
}


INSTANTIATE_LAYER_GPU_FUNCS(ExpandLabelLayer);

}  // namespace caffe
