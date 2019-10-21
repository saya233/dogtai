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
import org.deeplearning4j.optimize.listeners.CollectScoresIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.annotation.PostConstruct;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

class LRModel {

    private int EPOCH = -1;
    private int BATCH_SIZE = -1;
    private int dim = -1;
    MultiLayerNetwork model;
    Configuration config;

    public String init(String type, int dim,String path) throws IOException {

        config = new Configuration();
        config.setBoolean(SVMLightRecordReader.ZERO_BASED_INDEXING, false);
        config.setInt(SVMLightRecordReader.NUM_FEATURES, dim);
        this.dim = dim;
        if (type.equals("train")) {
            this.model = buildModel();
            return "start training model";
        } else {
            INDArray tmp = Nd4j.rand(1, dim);
            load(path);
            return "success";
        }
    }

    public MultiLayerNetwork buildModel() {
        OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                .nIn(dim)
                .nOut(2)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SIGMOID)
                .build();
        MultiLayerConfiguration logistic = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .list()
                .layer(0, outputLayer)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(logistic);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        return net;
    }

    public double fit(File file) throws IOException, InterruptedException {
        FileSplit train = new FileSplit(file);
        SVMLightRecordReader trainRecorder = new SVMLightRecordReader();
        trainRecorder.initialize(config,train);
        DataSetIterator iterator = new RecordReaderDataSetIterator(trainRecorder,128,dim,2);
        Double score = 0.0;
        int iter = 0;
        while (iterator.hasNext()){
            this.model.fit(iterator.next());
            score+=this.model.score();
            iter++;
        }
        return score/iter;
    }

    public void save(int epoch) throws  IOException{
        SimpleDateFormat format = new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");
        String modelname = format.format(new Date()).toString()+"_"+Integer.toString(epoch)+".zip";
        File modelfile = new File(modelname);
        this.model.save(modelfile,true);
    }

    public  void load(String name) throws IOException {
        File file = new File(name);
        this.model=MultiLayerNetwork.load(file,true);
    }
}