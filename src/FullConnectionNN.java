import java.util.List;

public class FullConnectionNN {

	// layer num, include input layer and ouput layer
	int layer_num;
	int total_unit_num;
	int total_par_num;
	int[] layer_units;
	ActivationFunc[] layer_activation_fun;
	double[] par;
	double[] history_par_grad_sum;

	// gradient checking a value buffer
	double[] checking_par_change_a_buf;

	// forward and back propagation buf
	double[] a_buf;
	double[] z_buf;
	double[] sigma_buf;
	double[] grad;
	double regular_lambda;

	// for compute
	int[] layer_par_num;
	int[] layer_w_disp; // layer w parameter displacement
	int[] layer_b_disp; // layer b parameter displacement
	int[] layer_unit_disp; // layer units displacement

	public int zero_grad() {
		for (int i = 0; i < total_par_num; i++) {
			grad[i] = 0;
		}
		return 0;
	}

	int init(List<Integer> layer_units_list,
			List<ActivationFunc> layer_activation_fun_list) {
		this.layer_units = new int[layer_units_list.size()];
		for (int i = 0; i < layer_units_list.size(); i++) {
			this.layer_units[i] = layer_units_list.get(i);
		}

		this.layer_activation_fun = layer_activation_fun_list
				.toArray(new ArctanFunc[0]);

		System.out
				.println("full connection construct activation function success!");
		this.layer_num = layer_units.length;
		this.layer_par_num = new int[layer_num - 1];

		for (int i = 0; i < layer_par_num.length; i++) {
			layer_par_num[i] = (layer_units[i] + 1) * layer_units[i + 1];
			total_par_num += layer_par_num[i];
			total_unit_num += layer_units[i];
		}
		// output layer
		total_unit_num += layer_units[layer_num - 1];

		par = new double[total_par_num];
		grad = new double[total_par_num];
		// init the history par gradient sum
		this.history_par_grad_sum = new double[total_par_num];
		// update the total par gradient delta
		layer_w_disp = new int[layer_num - 1];
		layer_b_disp = new int[layer_num - 1];
		layer_unit_disp = new int[layer_num];
		layer_w_disp[0] = 0;
		layer_b_disp[0] = layer_units[0] * layer_units[1];
		layer_unit_disp[0] = 0;
		for (int i = 1; i < layer_w_disp.length; i++) {
			layer_w_disp[i] += layer_w_disp[i - 1] + layer_par_num[i - 1];
			// !!IMPORTANT:
			layer_b_disp[i] = layer_w_disp[i] + layer_units[i]
					* layer_units[i + 1];
			layer_unit_disp[i] = layer_unit_disp[i - 1] + layer_units[i - 1];
		}
		layer_unit_disp[layer_num - 1] = layer_unit_disp[layer_num - 2]
				+ layer_units[layer_num - 2];

		a_buf = new double[total_unit_num];
		z_buf = new double[total_unit_num];
		sigma_buf = new double[total_unit_num];
		checking_par_change_a_buf = new double[total_unit_num];

		return 0;
	}

	

	int forward_propagation(double[] input) {
		if (input.length != layer_units[0]) {
			System.out
					.printf("error : [NeuralNetwork::forward_propagation] "
							+ "input vectore size(%d) != network layer 0 units num(%d)",
							input.length, layer_units[0]);
			return -1;
		}
		for (int i = 0; i < input.length; i++) {
			z_buf[i] = input[i];
			a_buf[i] = layer_activation_fun[0].activate(z_buf[i]);
		}
		for (int l = 1; l < layer_num; l++) {
			ActivationFunc act_fun = layer_activation_fun[l];
			int unit_disp = layer_unit_disp[l];
			int l_w_disp = layer_w_disp[l - 1];
			int l_b_disp = layer_b_disp[l - 1];
			for (int i = 0; i < layer_units[l]; i++) {
				// init z_value with bias
				z_buf[unit_disp + i] = par[l_b_disp + i];
				// for every output node,the weights connecting it is sequential
				int i_w_disp = l_w_disp + i * layer_units[l - 1];
				for (int j = 0; j < layer_units[l - 1]; j++) {
					z_buf[unit_disp + i] += par[i_w_disp + j]
							* a_buf[layer_unit_disp[l - 1] + j];
				}
				a_buf[unit_disp + i] = act_fun.activate(z_buf[unit_disp + i]);
			}
		}
		return 0;
	}

	int back_propagation(double []last_layer_grad) {

		if (grad.length != layer_units[layer_num - 1]) {
			System.out
					.printf("error : [NeuralNetwork::back_propagation] "
							+ " input grad size(%d) != neural network output layer units num(%d)",
							grad.length, layer_units[layer_num - 1]);
			return -1;
		}

		int output_unit_disp = layer_unit_disp[layer_num - 1];
		for (int i = 0; i < layer_units[layer_num - 1]; i++) {
			sigma_buf[output_unit_disp + i] = last_layer_grad[i]
					* layer_activation_fun[layer_num - 1]
							.gradient(z_buf[output_unit_disp + i]);
		}
		// compute sigma
		for (int l = layer_num - 2; l >= 0; l--) {
			int l_unit_disp = layer_unit_disp[l];
			int l_w_disp = layer_w_disp[l];
			ActivationFunc act_fun = layer_activation_fun[l];
			for (int i = 0; i < layer_units[l]; i++) {
				// init with zero
				sigma_buf[l_unit_disp + i] = 0;
				for (int j = 0; j < layer_units[l + 1]; j++) {
					sigma_buf[l_unit_disp + i] += par[l_w_disp + j
							* layer_units[l + 1] + i]
							* sigma_buf[layer_unit_disp[l + 1] + j];
				}

				// sigma[l_unit_disp + i] *= act_fun->grad(z[l_unit_disp + i]);
				// just on the activation value for sigmoid gradient f'(z) =
				// f(z)*(1-f(z)))
				sigma_buf[l_unit_disp + i] *= act_fun
						.gradient(z_buf[l_unit_disp + i]);
			}
		}

		// compute wij and b
		for (int l = 0; l < layer_num - 1; l++) {
			int l_w_disp = layer_w_disp[l];
			int l_b_disp = layer_b_disp[l];
			int l_unit_disp = layer_unit_disp[l];
			int ln_unit_disp = layer_unit_disp[l + 1];
			for (int i = 0; i < layer_units[l + 1]; i++) {
				for (int j = 0; j < layer_units[l]; j++) {
					grad[l_w_disp + i * layer_units[l] + j] += a_buf[l_unit_disp
							+ j]
							* sigma_buf[ln_unit_disp + i]
					 + regular_lambda * par[l_w_disp + i * layer_units[l] +
					 j];
				}
				grad[l_b_disp + i] += sigma_buf[ln_unit_disp + i];
			}
		}
		return 0;
	}

}
