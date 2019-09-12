import java.io.FileWriter;
import java.util.ArrayList;

public class Sin {
	public static void main(String argv[]) {
		try{
			 FileWriter fw = new FileWriter("sin_training_output.txt");
			  MLP NN2 = new MLP(4, 5, 1);
			  ArrayList<Vector> arraylist = new ArrayList<>();
			  for (int i=0;i<200;i++){
	               arraylist.add(new Vector());//creating 200 vectors and storing in list
			  }
			  
			  double maxEpochs = 10000;
	          double learningRate = 0.15;
	          
	          for (int e=0; e<maxEpochs; e++){
	                double error = 0;
	               
	                for (int p=0; p<150; p++) { //train on 1st 150 examples          
	                	Vector vector = arraylist.get(p);
	                	 NN2.forward(vector.getValues());
	                	 error += NN2.backwards(vector.getSin());
	                	 
	                }
	                NN2.updateWeights(learningRate);
	                fw.write("Error at epoch " + e + " is " + error + "\n");
	          }
	          fw.close();
	          
	          FileWriter fw2 = new FileWriter("sin_testing_output.txt");

	            for (int p=150; p<200; p++){ //test on last 50 examples
	                Vector vector = arraylist.get(p);
	                NN2.forward(vector.getValues());
	                fw2.write("Expected: " + vector.getSin()[0] + " Predicted: " + NN2.getOutput()[0] + "\n");
	            }
	          fw2.close();
			 
		}
		catch (Exception e)
        {
            e.printStackTrace();
        }
		
	}
}