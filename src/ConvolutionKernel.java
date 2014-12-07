
public class ConvolutionKernel {

	private int row_size;
	private int col_size;
	double kernel[];
	
	public ConvolutionKernel(int row_size,int col_size) {
		this.row_size = row_size;
		this.col_size = col_size;
		kernel = new double[row_size*col_size+1];
	}
	public double cov(FeatureMap fm,int s_row,int s_col)
	{
		double result = 0;
		int start_index_of_value = 0;
		for(int i = 0;i < row_size;i++){
			start_index_of_value = s_row * fm.row+s_col;
			for(int j = 0;j < col_size; j++){
				result += kernel[i*col_size+j]*fm.val[start_index_of_value+j];
			}
		}
		result += kernel[row_size*col_size];
		return result;
	}
}
