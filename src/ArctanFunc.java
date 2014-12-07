
public class ArctanFunc extends ActivationFunc {

	@Override
	protected double activate(double out) {
		return 2/Math.PI * Math.atan(out);
	}
	@Override
	protected double gradient(double out) {
		return 2/Math.PI /(1+(out*out));
	}
	@Override
	protected double gradient_on_a_value(double a_value) {
		 
		return 0;
	}
	
}
