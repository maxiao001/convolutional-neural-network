
public class SigmoidFunc extends ActivationFunc{

	@Override
	protected double activate(double out) {
		return 1/(1+Math.exp(out));
	}
	
	//can be optimized
	protected double gradient(double out){
		double acti_value = activate(out);
		return acti_value*(1-acti_value);
	}
	
	@Override
	protected double gradient_on_a_value(double a_value) {
		return a_value*(1+a_value);
	}
	
}
