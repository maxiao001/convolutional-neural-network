
public class LinearFunc extends ActivationFunc{

	@Override
	protected double activate(double out) {
		return out;
	}

	@Override
	protected double gradient(double out) {
		return 1;
	}

	@Override
	protected double gradient_on_a_value(double a_value) {
		return 1;
	}

}
