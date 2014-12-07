import java.util.ArrayList;

public class CNN {

	double learn_rate = 0.0001;
	double regular_lambda = 0;
	int max_iteration = 10000;
	
	public ArrayList<CNNLayer> layers;
	public ArrayList<FeatureMap[]> featureMap;
	public FullConnectionNN nn;

	public void addLayer(CNNLayer layer) {
		this.layers.add(layer);
	}

	public void setConfig(double learn_rate, double regular_lambda, int max_iteration) {
		this.learn_rate = learn_rate;
		this.regular_lambda = regular_lambda;
		this.max_iteration = max_iteration;
	}

	public void train_init() {
		init_cnn_layers();
		init_feature_map();
	}

	/**
	 * init variables of each layer,include divide the layers into convolution
	 * layer and full connection
	 */
	private void init_cnn_layers() {
		
	}

	/**
	 * init feature map of each layer
	 */
	private void init_feature_map() {

	}

	public void train(ArrayList<Sample> train_samples){
		for(int i = 0;i < max_iteration;i++)
		{
			int index = (int)Math.random() * train_samples.size();
			this.feed_forward(train_samples.get(index).value);
			this.compute_gradient(train_samples.get(index).label);
			this.back_propagation();
		}
		
	}


	private void feed_forward(double[] value) {
		
	}
	/**
	 * softmax classfication(multiple logistisic regression ) or linear regression  etc
	 * @param label
	 */
	private void compute_gradient(double label) {
		
	}
	
	private void back_propagation() {
		
	}
	
	public void predict(ArrayList<Sample> test_samples) {

	}

	public void save_model() {

	}

}
