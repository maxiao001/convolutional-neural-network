
public class FullConnectionLayer extends CNNLayer {

	//because the convolution one-one connects the previous feature maps and next feature maps
	ConvolutionKernel[] kernel;

	
	int unit_num;
	public FullConnectionLayer(int unit_num,ActivationFunc  act_fun) {
		this.unit_num = unit_num;
		this.act_fun = act_fun;
	}
	
	

}
