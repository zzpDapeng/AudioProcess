import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class Dictionary {
    private static final HashMap<String, Integer> word2index = new HashMap<>();
    private static final HashMap<Integer, String> index2word = new HashMap<>();

    public static void init(String dictionary_path) {
        try {
            BufferedReader in = new BufferedReader(new FileReader(dictionary_path));
            String str;
            while ((str = in.readLine())!= null){
                String[] items = str.split(" ");
                String word = items[0];
                int index = Integer.parseInt(items[1]);
                word2index.put(word,index);
                index2word.put(index, word);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static int word_to_index(String word) {
        return word2index.getOrDefault(word, 0);
    }

    public static String index_to_word(int index) {
        return index2word.getOrDefault(index, " ");
    }

    public static long[] Long_to_long(List<Long> arrayList){
        long[] list = new long[arrayList.size()];
        for (int i = 0; i < arrayList.size(); i++) {
            list[i] = arrayList.get(i);
        }
        return list;
    }

    public static float[][][] clip_feature(float[][] audio_feature) {
        int MAX_AUDIO_FEATURE_LEN = 400;
        int feature_dim = audio_feature[0].length;
        int clip_num = (int) Math.ceil(audio_feature.length / (float) MAX_AUDIO_FEATURE_LEN);
        float[][][] temp = new float[clip_num][MAX_AUDIO_FEATURE_LEN][feature_dim];
        for (int i = 0; i < clip_num; i++) {
            System.arraycopy(audio_feature, MAX_AUDIO_FEATURE_LEN*i, temp[i], 0, clip_num);
        }
        return temp;
    }

    public static void main(String[] args) {
//        init("src/grapheme_table.txt");
//        System.out.println(word_to_index("你"));
//        System.out.println(index_to_word(1));
        ArrayList<Long> arrayList = new ArrayList<>();
        arrayList.add((long)1);
        long[] list = Long_to_long(arrayList);
        for (long l : list) {
            System.out.println(l);
        }
    }
}
