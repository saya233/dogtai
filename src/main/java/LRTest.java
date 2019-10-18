import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

public class LRTest {
    public static void main(String[] args) throws InterruptedException {
        Random random = new Random();
        SparseLR lr = new SparseLR();
        System.out.println(lr.init());


        long a = System.currentTimeMillis();
        double [] [] tests= new double[10][123];
        for (int i = 0; i <10 ; i++) {
            for (int j = 0; j < 123; j++) {
                tests[i][j] = random.nextDouble();
            }
        }
        INDArray predict = lr.predict(tests);
        System.out.println(System.currentTimeMillis()-a);

        Thread.sleep(3000);
        long b = System.currentTimeMillis();
        double [] [] tt= new double[10][123];
        for (int i = 0; i <10 ; i++) {
            for (int j = 0; j < 123; j++) {
                tt[i][j] = random.nextDouble();
            }
        }
        INDArray p = lr.predict(tt);
        System.out.println(System.currentTimeMillis()-b);
    }
}
