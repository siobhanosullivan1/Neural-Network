import java.io.FileWriter;
import java.util.ArrayList;

public class XOR {
	public static void main(String argv[]) {
		try{
			 FileWriter fw = new FileWriter("xor_output.txt");
			 MLP NN = new MLP(2,2,1);
			 double[][] input = {{0, 0},{0, 1},{1, 0},{1, 1}};
			 double[][] output = {{0},{1},{1},{0}};
			 int maxEpochs = 10000;
	         int numExamples = input.length;
	         double learningRate = 0.15;
	         
	         for(int e=0; e<maxEpochs; e++){
	        	 double error = 0; ;
	        	 for (int p=0; p<numExamples; p++) {
	        		 NN.forward(input[p]);
	        		 error += NN.backwards(output[p]);
	        		 
	        		 if (p%2 == 0){ //updates weights every now and then
	                        NN.updateWeights(learningRate);
	        		 }
	        		 
	        		 fw.write("Expected is " + output[p][0] + " and predicted is " + NN.getOutput()[0] + "\n");
	        		 
	        	 }
	        	 fw.write("Total error at epoch " + e + " is " + error + "\n");
	         }
	         fw.close();

		}
		 catch (Exception e)
        {
            e.printStackTrace();
        }
		
		
		
		
		
	}
}
