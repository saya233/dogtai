import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.annotation.PostConstruct;
import java.io.File;
import java.io.IOException;
import java.util.Random;

class SparseLR{

    private int EPOCH = -1;
    private int BATCH_SIZE = -1;

    MultiLayerNetwork model;

    @PostConstruct
    public String init() {

        INDArray tmp = Nd4j.rand(1,123);
        try{
            this.model=loadModel();
            this.model.output(tmp);
            return "success";
        }catch (Exception e){
            return "load model failed";
        }
    }

    public MultiLayerNetwork loadModel() throws IOException {
        File locationToSave = new File("lrModel.zip");
        boolean saveUpdater = true;
        MultiLayerNetwork model = MultiLayerNetwork.load(locationToSave,true);
        return model;
    }

    public  INDArray predict(double [][] data){
        long a = System.currentTimeMillis();
        INDArray d = Nd4j.create(data);
        long b = System.currentTimeMillis();
        INDArray ans = this.model.output(d,false);
        long c = System.currentTimeMillis();
        return ans;
    }


    public INDArray predict(INDArray samples){
        return model.output(samples);
    }

}