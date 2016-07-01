public class Synapse {
	Neuron in; // The neuron sending the information.
	Neuron out; // The neuron receiving the information.
	double weight;

	/**
	 * Constructs a synapse between the IN neuron to the OUT neuron; the former sends information forward to the latter.
	 * @param in The neuron in the previous layer.
	 * @param out The neuron in the next layer.
	 */
	public Synapse(Neuron in, Neuron out) {
		this.in = in;
		this.out = out;
		in.downstreamSynapses.add(this);
		out.incomingSynapses.add(this);
		weight = (Math.random() - 0.5) * 2.0;
	}
}