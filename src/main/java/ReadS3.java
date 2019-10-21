import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.misc.SVMLightRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;

public class ReadS3 {
    public static void main(String[] args) throws IOException, InterruptedException {
        int EPOCH = 30;
        double score =0;
        LRModel model = new LRModel();
        String state = model.init("test",71,"model/2_model.zip");
        System.out.println(state);
        File file = new File("../alldata");
        for (int i = 0; i <EPOCH ; i++) {
            score = model.fit(file);
            model.save(i);
            File logfile = new File("log.data");
            String log = "Epoch " + Integer.toString(i) + " loss " + Double.toString(score);
            PrintWriter printWriter = new PrintWriter(new FileWriter(logfile, true), true);
            printWriter.println(log);
            printWriter.close();
        }
//
//        File trainFileList = new File(path); 
//        double score = 0;
//         LRModel model = new LRModel();
//         String state = model.init("train", 71); 
//        System.out.println(state); 
//        for (int i = 0; i < EPOCH; i++)
//        { 
//            long a = System.currentTimeMillis(); 
//            for (File file : trainFileList.listFiles())
//            { 
//                System.out.println(file.getName());
//                 score = model.fit(file); 
//            } 
//            model.save(i);
//             File logfile = new File("log.data"); 
//            String log = "Epoch " + Integer.toString(i) + " loss " + Double.toString(score); 
//            PrintWriter printWriter = new PrintWriter(new FileWriter(logfile, true), true); printWriter.println(log); 
//            printWriter.close(); 
//            System.out.println("Epoch " + Integer.toString(i) + " " + Long.toString(System.currentTimeMillis() - a)); }  
//        String ans = model.init("train", 71); System.out.println(ans);

//
//        inputStream.close();
//        bufferedReader.close();
    }
}
