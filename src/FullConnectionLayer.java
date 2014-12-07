
public class FullConnectionLayer extends CNNLayer {

	//because the convolution one-one connects the previous feature maps and next feature maps
	ConvolutionKernel[] kernel;

	int prev_feature_map_num;
	int current_feature_map_num;
	int kernel_num = 0;
	int kernel_row_size = 0;
	int kernel_col_size = 0;
	
	public FullConnectionLayer(int kernel_num, int kernel_row_size,
			int kernel_col_size,ActivationFunc  act_fun) {
		this.kernel_num = kernel_num;
		this.kernel_row_size = kernel_row_size;
		this.kernel_col_size = kernel_col_size;
		this.act_fun = act_fun;
	}
	

}
