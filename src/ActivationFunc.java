
public abstract class ActivationFunc {

	protected abstract  double activate(double out);
	protected abstract double gradient(double out);
	protected abstract double gradient_on_a_value(double a_value);
}
