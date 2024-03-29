
package hillclimbing;

import java.util.stream.IntStream;

public class Layer {
    static enum LayerType {//input hidden output
        I, H,H2, O};
    private Neuron[] neurons=null;
    private LayerType layerType;
    public Layer(NeuralNetwork neuralNetwork,LayerType layerType){
            this.layerType=layerType;
            switch(layerType){
            case I:
                neurons=new Neuron[NeuralNetwork.NUMB_OF_INPUT_NEURONS];
                IntStream.range(0, NeuralNetwork.NUMB_OF_INPUT_NEURONS).forEach(i->neurons[i]=new Neuron(layerType,0));
                break;
            case H:
                 neurons=new Neuron[neuralNetwork.getNumbOfHiddenNeurons()];
                IntStream.range(0, neuralNetwork.getNumbOfHiddenNeurons()).forEach(i->neurons[i]=new Neuron(layerType,NeuralNetwork.NUMB_OF_INPUT_NEURONS));
                break;
           case H2://加上
                 neurons=new Neuron[neuralNetwork.getNumbOfHiddenNeurons()];
                IntStream.range(0, neuralNetwork.getNumbOfHiddenNeurons()).forEach(i->neurons[i]=new Neuron(layerType,8));
                break;      

            case O:
                neurons=new Neuron[NeuralNetwork.NUMB_OF_OUTPUT_NEURONS];
                neurons[0]=new Neuron(layerType,neuralNetwork.getNumbOfHiddenNeurons());
                break; 
        }
    }
    
    
    public Neuron[] getNeuron(){ return neurons;}
    public LayerType getLayerType(){return layerType;}
    public String toString(){
            StringBuffer returnValue=new StringBuffer();
            IntStream.range(0, neurons.length).forEach(x->returnValue.append(neurons[x]+" "));
            return returnValue.toString();
    }
}
