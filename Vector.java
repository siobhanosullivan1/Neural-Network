import java.util.concurrent.ThreadLocalRandom;

public class Vector {
	
    private double[] values; 
    
    public Vector(){ //input vectors which will contain a random number between -1 and 1
    	values = new double[4];
    	for(int i=0; i<4; i++){
    		values[i] = ThreadLocalRandom.current().nextDouble(-1, 1);
    	}
    }
    
    public double[] getValues(){
    	return values;
    }
    
    public double[] getSin(){
    	double[] output = {Math.sin(values[0] - values[1] + values[2] - values[3])};
    	return output;
    }
        
}
