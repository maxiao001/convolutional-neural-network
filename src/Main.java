import java.util.ArrayList;


public class Main {

	public static void main(String[] args) {

		CNN cnn = new CNN();
		ActivationFunc sigmoid_act = new SigmoidFunc();
		
		//suppose the input is 28*28,make sure that the feature map  still > 1*1 during layers train
		CNNLayer layer1 = new ConvolutionLayer(6, 5, 5, sigmoid_act);//24*24
		CNNLayer layer2 = new SubsampleLayer(6, 2, 2, sigmoid_act);//12*12
		CNNLayer layer3 = new ConvolutionLayer(12, 5, 5, sigmoid_act);//8*8
		CNNLayer layer4 = new SubsampleLayer(12, 2, 2, sigmoid_act);//4*4
		CNNLayer layer5 = new ConvolutionLayer(20, 4, 4, sigmoid_act);//1*1  20 feature map
		CNNLayer layer6 = new FullConnectionLayer(15,sigmoid_act);//full nn  hidden layer
		CNNLayer layer7 = new FullConnectionLayer(10,sigmoid_act);//full nn output layer for multiple class classification
		cnn.addLayer(layer1);
		cnn.addLayer(layer2);
		cnn.addLayer(layer3);
		cnn.addLayer(layer4);
		cnn.addLayer(layer5);
		cnn.addLayer(layer6);
		cnn.addLayer(layer7);
		
		ArrayList<Sample> train_samples = load_train_samples();
		ArrayList<Sample> test_samples = load_test_samples();
		
		cnn.train_init();
		double learn_rate = 0.0001;
		double regular_lambda = 0;
		int max_iteration = 100000;
		cnn.setConfig(learn_rate,regular_lambda,max_iteration);
		cnn.train(train_samples);
		cnn.predict(test_samples);
		cnn.save_model();
	}

	private static ArrayList<Sample> load_train_samples() {
		return null;
	}
	private static ArrayList<Sample> load_test_samples() {
		return null;
	}
	

}
