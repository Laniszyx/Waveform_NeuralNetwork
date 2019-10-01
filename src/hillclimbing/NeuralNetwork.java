package hillclimbing;

import com.sun.org.apache.xerces.internal.util.IntStack;
import java.util.logging.Logger;
import java.util.stream.IntStream;

public class NeuralNetwork {
    static final double LEARNING_RATE=0.8;
    final static int NUMB_OF_INPUT_NEURONS =Driver.TRAINING_DATA[0][0].length;
     final static int NUMB_OF_OUTPUT_NEURONS = 1;
     private int numbOfHiddenNeurons;
     private Layer[] layers=new Layer[Layer.LayerType.values().length];
     public NeuralNetwork(int numbOfHiddenNeurons){
            this.numbOfHiddenNeurons=numbOfHiddenNeurons;
            layers[0]=new Layer(this,Layer.LayerType.I);
            layers[1]=new Layer(this,Layer.LayerType.H);
            layers[2]=new Layer(this,Layer.LayerType.H2);
            layers[3]=new Layer(this,Layer.LayerType.O);
     }
     
     public NeuralNetwork forwardprop(double input[]){
         for(int i=0;i<layers.length;i++){ 
             switch(layers[i].getLayerType()){
         case I:
                for(int j=0;j<layers[i].getNeuron().length;j++)
                layers[i].getNeuron()[j].setOutput(input[j]);
                break;
            case H:
               for(int j=0;j<layers[i].getNeuron().length;j++){
                   double weightedSum=0;
                     for(int k=0;k<layers[i].getNeuron()[0].getWeights().length;k++)
                         weightedSum+=layers[i].getNeuron()[j].getWeights()[k]*layers[i-1].getNeuron()[k].getOutput();
                          layers[i].getNeuron()[j].applyActivationFunction(weightedSum);
               }
                break;
                
              case H2:
               for(int j=0;j<layers[i].getNeuron().length;j++){
                   double weightedSum=0;
                     for(int k=0;k<layers[i].getNeuron()[0].getWeights().length;k++)
                         weightedSum+=layers[i].getNeuron()[j].getWeights()[k]*layers[i-1].getNeuron()[k].getOutput();
                          layers[i].getNeuron()[j].applyActivationFunction(weightedSum);
               }
                break;   
                
                
                
            case O:
             double weightedSum=0;
                     for(int k=0;k<layers[i].getNeuron()[0].getWeights().length;k++)
                         weightedSum+=layers[i].getNeuron()[0].getWeights()[k]*layers[i-1].getNeuron()[k].getOutput();
                          layers[i].getNeuron()[0].applyActivationFunction(weightedSum);
                break;
             }
         } 
         return this;
     }
     
     public NeuralNetwork testforwardprop(double input[]){
       double[] weightsH1=new double[]{0.01, 0.51, 0.01, -0.59};
        double[] weightsH2=new double[]{  0.33, -0.04, -0.37, -0.31};
        double[] weightsH3=new double[]{ -0.37, -0.48, 0.04, 0.34};
        double[] weightsH4=new double[]{   0.76, 0.40, -0.22, 0.10 };
        double[] weightsH5=new double[]{ 0.81, -0.15, -0.16, -0.28};
        double[] weightsH6=new double[]{  0.30, 0.00, 0.31, 0.35  };
        double[] weightsH7=new double[]{  0.77, 0.48, 0.07, 0.06};
        double[] weightsH8=new double[]{ 0.11, -0.02, 0.12, -0.29 };
        double[] weightsO1=new double[]{  1.36, 1.15, 0.47, 1.78, 1.67, 0.23, 1.63, 0.52 };  
         for(int i=0;i<layers.length;i++){ 
             switch(layers[i].getLayerType()){
         case I:
                for(int j=0;j<layers[i].getNeuron().length;j++)
                layers[i].getNeuron()[j].setOutput(input[j]);
                break;
            case H:
               for(int j=0;j<4;j++){
                   double weightedSum=0;
                     for(int k=0;k<4;k++){
                         switch(j){
                             case 0:
                         weightedSum+=weightsH1[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 1:
                         weightedSum+=weightsH2[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 2:
                         weightedSum+=weightsH3[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 3:
                         weightedSum+=weightsH4[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 4:
                         weightedSum+=weightsH5[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 5:
                         weightedSum+=weightsH6[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 6:
                         weightedSum+=weightsH7[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 7:
                         weightedSum+=weightsH8[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                         }
                layers[i].getNeuron()[j].applyActivationFunction(weightedSum);}
               }
                break;
            case O:
             double weightedSum=0;
                     for(int k=0;k<8;k++){
                          switch(k){
                         case 0:
                         weightedSum+=weightsO1[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 1:
                         weightedSum+=weightsO1[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 2:
                         weightedSum+=weightsO1[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 3:
                         weightedSum+=weightsO1[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 4:
                         weightedSum+=weightsO1[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 5:
                         weightedSum+=weightsO1[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 6:
                         weightedSum+=weightsO1[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          case 7:
                         weightedSum+=weightsO1[k]*layers[i-1].getNeuron()[k].getOutput();
                         break;
                          }
                           layers[i].getNeuron()[0].applyActivationFunction(weightedSum);
                     }
                break;
             }
         } 
         return this;
     }
     
     
      public NeuralNetwork backpropError(double targetResult){
          Neuron[] iNeuron=layers[0].getNeuron();
         Neuron[] hNeuron=layers[1].getNeuron();
         Neuron[] h2Neuron=layers[2].getNeuron();
         Neuron oNeuron=layers[layers.length-1].getNeuron()[0];
         oNeuron.setError((targetResult-oNeuron.getOutput())*oNeuron.derivation());
         for(int j=0;j<oNeuron.getWeights().length;j++)
             oNeuron.getWeights()[j]=oNeuron.getWeights()[j]+LEARNING_RATE*oNeuron.getError()*h2Neuron[j].getOutput();
         
         for(int i=0;i<h2Neuron.length;i++){
             h2Neuron[i].setError((oNeuron.getWeights()[i]*oNeuron.getError())*h2Neuron[i].derivation());
             for(int j=0;j<h2Neuron[0].getWeights().length;j++)
             h2Neuron[i].getWeights()[j]=h2Neuron[i].getWeights()[j]+LEARNING_RATE*h2Neuron[i].getError()*hNeuron[j].getOutput();
         }

         for(int i=0;i<hNeuron.length;i++){
             hNeuron[i].setError((h2Neuron[i].getWeights()[i]*h2Neuron[i].getError())*hNeuron[i].derivation());
             for(int j=0;j<hNeuron[0].getWeights().length;j++)
             hNeuron[i].getWeights()[j]=hNeuron[i].getWeights()[j]+LEARNING_RATE*hNeuron[i].getError()*iNeuron[j].getOutput();
         }
         return this;
     }


    public int getNumbOfHiddenNeurons() {
        return numbOfHiddenNeurons;
    }

    public Layer[] getLayers() {
        return layers;
    }
    
    public String toString(){
        StringBuffer returnValue=new StringBuffer();
        IntStream.range(0, layers.length).forEach(x->returnValue.append(layers[x]+"   "));
        return returnValue.toString();
    }
 
   
    
}
