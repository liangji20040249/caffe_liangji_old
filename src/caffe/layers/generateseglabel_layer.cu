/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/generateseglabel_layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype get_data(Dtype * data, int num, int channels,int height, int width,int n,int c,int h,int w,float v)
{
	if(h<0 || h >=height)
		return Dtype(v);
	if(w<0 || w >= width)
		return Dtype(v);
	
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
__global__ void get_seglabel(const int count, const Dtype* bottom_data,Dtype * top1_data,Dtype * top2_data,Dtype * top3_data,Dtype * top4_data,Dtype * top5_data,
				int num,int in_height,int in_width,int out_channels,int out_height,int out_width,int ignore_label,int seg_ignore_range,
				int reg_range,float resize_ratio,int seg_label_shift) {
CUDA_KERNEL_LOOP(index, count) {


	int temp=index;

	int n,c,h,w;
	n = temp / (out_height * out_width);
	temp = temp % (out_height * out_width);
	h = temp / out_width;
	temp = temp % out_width;
	w = temp;
	
	int sn,sc,sh,sw;
	sn=n;
	sc=0;
	
	sh = float(h) * resize_ratio;
	sw = float(w) * resize_ratio;

	bool isedge=false;
	//Dtype srcv = get_data(bottom_data, num, 1,in_height, in_width,sn,sc,sh,sw,float(ignore_label));
	//Dtype srcv = bottom_data[sn*1*in_height*in_width + sc * in_height*in_width + sh * in_width + sw];
	
	int idx = sn*1*in_height*in_width + sc * in_height*in_width + sh * in_width + sw;
	Dtype srcv = ignore_label;
	if(sw<0||sw>=in_width||sh<0||sh>=in_height)
	{
		;
	}
	else if(idx>=0 && idx < num * 1 * in_height * in_width)
		srcv = bottom_data[idx];


	if(int(srcv)==ignore_label)
	{
		set_data(top2_data, num, 1,out_height, out_width, n, 0, h, w,Dtype(0));
		return;
	}

	Dtype nebv;
	float min_dist = reg_range*2;
	float min_dx,min_dy;
	float dist;
	for(int i=-reg_range;i<=reg_range;i++)
	{
		for(int j=-reg_range;j<=reg_range;j++)
		{
			
if(i==0 && j ==0)
			{
				continue;
			}
			int x=sw+i;
			int y=sh+j;
			if(x<0||x>=in_width||y<0||y>=in_height)
			{
				continue;
			}
			
			nebv = srcv;
			idx = sn*1*in_height*in_width + sc * in_height*in_width + y * in_width + x;
			if(idx>=0 && idx < num * in_height * in_width)
				nebv = bottom_data[idx];
			

			//nebv = get_data(bottom_data, num, 1,in_height, in_width,sn,sc,y,x,float(srcv));
			//nebv = bottom_data[sn*1*in_height*in_width + sc * in_height*in_width + y * in_width + x];
			if(int(srcv)!=int(nebv))
			{
				isedge = true;
				dist = sqrt(float(i*i + j*j));
				if(dist < min_dist)
				{
					min_dist = dist;
					min_dx = i;
					min_dy = j;
				}
			}
		}
	}


	
	//set expand seg label
	int lb = (int)srcv - seg_label_shift;
	if(lb>=0 && lb < out_channels)
	{
	
		//top1_data[0]=1.0;
		//top1_data[n*out_channels*out_height*out_width + lb * out_height*out_width + h * out_width + w]=1;
		//top1_data[n*out_channels*out_height*out_width + 0 * out_height*out_width + h * out_width + w]=1;
		
		set_data(top1_data, num, out_channels,out_height, out_width, n, lb, h, w,Dtype(1));
	}

	//set seg ignore
	if(isedge && min_dist <= seg_ignore_range)
	{
		set_data(top2_data, num, 1,out_height, out_width, n, 0, h, w,Dtype(0));
	}
	/*if(int(srcv) == ignore_label)
	{
		set_data(top2_data, num, 1,out_height, out_width, n, 0, h, w,Dtype(0));
	}*/

	//set reg and mask
	if(isedge && min_dist <= reg_range)
	{
		set_data(top3_data, num, 1,out_height, out_width, n, 0, h, w,Dtype(min_dist));
		set_data(top5_data, num, 2,out_height, out_width, n, 0, h, w,Dtype(min_dx));
		set_data(top5_data, num, 2,out_height, out_width, n, 1, h, w,Dtype(min_dy));
		set_data(top4_data, num, 1,out_height, out_width, n, 0, h, w,Dtype(1));	
	}


}
}

template <typename Dtype>
void GenerateSeglabelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    


	//H_.Reshape(num_, channels_, height_, width_);
	//caffe_copy(H_.count(),bottom[0]->gpu_data(),H_.mutable_gpu_data());
	//const Dtype* bottom_data = H_.gpu_data();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	//Dtype* bottom_data = bottom[0]->mutable_gpu_data();
	Dtype* top1_data = top[0]->mutable_gpu_data();
	Dtype* top2_data = top[1]->mutable_gpu_data();
	Dtype* top3_data = top[2]->mutable_gpu_data();
	Dtype* top4_data = top[3]->mutable_gpu_data();
	Dtype* top5_data = top[4]->mutable_gpu_data();
	

	caffe_gpu_set(top[0]->count(), Dtype(0),top1_data);
	caffe_gpu_set(top[1]->count(), Dtype(1),top2_data);
	caffe_gpu_set(top[2]->count(), Dtype(0),top3_data);
	caffe_gpu_set(top[3]->count(), Dtype(0),top4_data);
	caffe_gpu_set(top[4]->count(), Dtype(0),top5_data);

	const int count = num_ * out_height_ * out_width_;

	
	get_seglabel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,bottom_data,top1_data,top2_data,top3_data,top4_data,top5_data,num_,height_,width_,out_channels_,out_height_,out_width_,ignore_label_,seg_ignore_range_,reg_range_,resize_ratio_,seg_label_shift_);

	CUDA_POST_KERNEL_CHECK;
	


}

template <typename Dtype>
void GenerateSeglabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    	;

}

INSTANTIATE_LAYER_GPU_FUNCS(GenerateSeglabelLayer);

}  // namespace caffe
