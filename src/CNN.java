import java.util.ArrayList;


public class CNN {

	public ArrayList<CNNLayer> layers;
	public ArrayList<FeatureMap[]> featureMap;
	public FullConnectionNN nn;
	
	public void addLayer(CNNLayer layer){
		this.layers.add(layer);
	}
	public void train_init(){
		init_cnn_layers();
		init_feature_map();
	}
	
	private void init_cnn_layers() {
		
	}
	private void init_feature_map() {
		
	}
	public void ff(){
		
	}
	public void bp(){
		
	}
	
}

