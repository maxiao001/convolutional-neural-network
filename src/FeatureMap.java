
public class FeatureMap {

	int row;
	int col;
	double val[];
	double label;
	
	private FeatureMap(int row_size,int col_size){
		this.row = row_size;
		this.col = col_size;
		this.val = new double[row_size*col_size];
	}
	public void setVal(double value[])
	{
		if(this.val.length != value.length){
			System.err.println("the val and input values length not match");
		}
		for(int i = 0; i < value.length;i++){
			val[i] = value[i];
		}
	}
		
}
