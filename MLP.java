import java.util.Random;

public class MLP {

	private int NI;
	private int NH;
	private int NO;
	private double w1[][]; //array with weights in lower layer
	private double w2[][]; //array with weights in higher layer
	private double dw1[][]; //array with weight changes to be applied to w1 and w2
	private double dw2[][];
	private double z1[]; //array with activation for lower layer
	private double z2[]; //array with activation for higher layer
	private double h[]; //where the values of the hidden neurons are stored
	private double o[]; //stores outputs
	
	private double inputs[];
	
	public MLP(int numInputs, int numHidden, int numOutputs){
		NI = numInputs;
		NH = numHidden;
		NO = numOutputs;
		
		w1 = new double[NI][NH];
		w2 = new double[NH][NO];
		
		dw1 = new double[NI][NH];
		dw2 = new double[NH][NO];
		
		z1 = new double[NH];
		z2 = new double[NO];
		
		h = new double[NH];
		o = new double[NO];
		
		w1 = randomise(w1);
		w2 = randomise(w2);
		dw1 = setToZeroes(dw1);
		dw2 = setToZeroes(dw2);
		
	}
	
	public double[][] randomise(double[][] a){ //initializes w1 and w2 to small random values
	//set dw1 and dw2 to zeroes??
		Random rand = new Random();
		double[][] x = new double[a.length][a[0].length]; //creates an array same size and dimensions as the one taken in
		
		for(int i=0; i<a.length; i++){
			for(int j=0; j<a[0].length; j++){
				x[i][j] = rand.nextDouble(); //generates random number between 0 and 1 
			}
		}		
		return x;
	}
	
	public double[][] setToZeroes(double[][] a){ //for setting dw1 and dw2 to all zeroes
		double[][] x = new double[a.length][a[0].length]; 
		
		for(int i=0; i<a.length; i++){
			for(int j=0; j<a[0].length; j++){
				x[i][j] = 0;
			}
		}		
		return x;
	}
	
	public void forward(double[] input){
		inputs = input;
		
		for(int i=0; i<NH; i++){ //calculates the hidden values
			for(int j=0; j<NI; j++){
				z1[i] = inputs[j]*w1[j][i];
				h[i] = (1.0/1.0 + Math.exp(-z1[i]));
			}	
		}	
		
		for(int k=0; k<NO; k++){ //calculates the output
			for(int l=0; l<NI; l++){
				 z2[k] =z1[l]*w2[l][k]; //activation for output
				 o[k] = (1.0/1.0 + Math.exp(-z2[k]));
			}
		}
		
	}

	public double backwards(double[] target){
		double delta=0;
		
		for(int i=0; i<NH; i++){ //upper layer
			for(int j=0; j<NO; j++){
						
				double sigmoidDerivative = (1.0/1.0 + Math.exp(-z2[j])) * (1.0-(1.0/1.0 + Math.exp(-z2[j])));
				delta = getMeanSquaredError(target[j], o[j])* sigmoidDerivative;
				dw2[i][j] += delta*h[j]; //deltas * hidden values
				
			}
		}
		for(int i=0; i<NI; i++){ //lower layer
			for(int j=0; j<NH; j++){
				
				double hidden = delta*w2[j][0];
				double sigmoidDerivative = (1.0/1.0 + Math.exp(-z1[j])) * (1.0-(1.0/1.0 + Math.exp(-z1[j])));
				double deltaHidden = hidden * sigmoidDerivative;
				dw1[i][j] += deltaHidden * inputs[i]; //deltas * inputs
			}
		}
		return getError(target);
	}
	
	public void updateWeights(double learningRate){
		
		for(int i=0; i<NI; i++){
			for(int j=0; j<NH; j++){
				w1[i][j] += learningRate*dw1[i][j];
				
			}
		}
		for(int i=0; i<NH; i++){
			for(int j=0; j<NO; j++){
				w2[i][j] += learningRate*dw2[i][j];
				
			}
		}
		dw1 = setToZeroes(dw1);
		dw2 = setToZeroes(dw2);
	}
	
	public double getError(double[] target){ //returns error on the example
		double sum=0;
		for(int i=0; i<NO; i++){
			sum += getMeanSquaredError(target[i], o[i]); 
		}
		return sum;
	}
	
	public double getMeanSquaredError(double target, double actual){		
		return 0.5*(Math.pow(target-actual,2));
	}
	
	public double[] getOutput(){
		return o;
	}

	
	
	
	
	
	
	
	
	}
