import java.util.Arrays;

public class AudioProcess {

    /**
     * read audio data from audio file
     *
     * @param file_path:file path
     * @return : 1-dim float audio data
     */
    public static float[] read_wav_from_file(String file_path) {
        WaveFileReader reader = new WaveFileReader(file_path);
        int[][] data3 = reader.getData();
        float[][] data4 = new float[data3.length][data3[0].length];
        for (int m = 0; m < data3.length; m++) {
            for (int n = 0; n < data3[0].length; n++) {
                data4[m][n] = (float) data3[m][n];
            }
        }
        return data4[0];
    }

    public static short[] read_wav_short(String file_path) {
        WaveFileReader reader = new WaveFileReader(file_path);
        int[][] data3 = reader.getData();
        short[][] data4 = new short[data3.length][data3[0].length];
        for (int m = 0; m < data3.length; m++) {
            for (int n = 0; n < data3[0].length; n++) {
                data4[m][n] = (short) data3[m][n];
            }
        }
        return data4[0];
    }

    /**
     * transpose a 2-dim matrix
     *
     * @param data: 2-dim matrix
     * @return ： float[][], transposed 2-dim matrix
     */
    public static float[][] T(float[][] data) {
        float[][] temp = new float[data[0].length][data.length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                temp[j][i] = data[i][j];
            }
        }
        return temp;
    }


    /**
     * log operation on a 2-dim matrix
     *
     * @param data: a 2-dim matrix
     * @return : logged 2-dim matrix
     */
    public static float[][] log(float[][] data) {
        float[][] temp = new float[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                if (data[i][j] != 0) {
                    temp[i][j] = (float) Math.log(data[i][j]);
                } else {
                    temp[i][j] = 0;
                }
            }
        }
        return temp;
    }

    /**
     * 特征堆叠
     *
     * @param data          ： 音频特征
     * @param left_context  ：左边特征数量
     * @param right_context ：右边特征数量
     * @return ：堆叠后的音频特征
     */
    public static float[][] concat_frame(float[][] data, int left_context, int right_context) {
        int time_steps = data.length;
        int feature_dim = data[0].length;
        float[][] concat_feature = new float[time_steps][feature_dim * (1 + left_context + right_context)];
        for (int i = 0; i < time_steps; i++) {
            for (int j = 0; j < feature_dim; j++) {
                for (int k = 0; k <= left_context; k++) {
                    if (i + k < time_steps) {
                        concat_feature[i + k][j + (left_context - k) * feature_dim] = data[i][j];
                    } else {
                        concat_feature[(i + k) % time_steps][j + (left_context - k) * feature_dim] = 0;
                    }
                }
                for (int k = 1; k <= right_context; k++) {
                    if (i - k >= 0) {
                        concat_feature[i - k][j + (left_context + k) * feature_dim] = data[i][j];
                    } else {
                        concat_feature[i - k + time_steps][j + (left_context + k) * feature_dim] = 0;
                    }
                }
            }
        }
        return concat_feature;
    }


    /**
     * 音频特征下采样
     *
     * @param data         ： 音频特征
     * @param sample_ratio ： 采样率
     * @return ：下采样后的音频特征
     */
    public static float[][] subsampling(float[][] data, int sample_ratio) {
        int frame_num = (int) Math.ceil(data.length / (float) sample_ratio);
        float[][] subsampling_feature = new float[frame_num][data[0].length];
        for (int i = 0; i < data.length; i++) {
            if (i % sample_ratio == 0) {
                System.arraycopy(data[i], 0, subsampling_feature[i / sample_ratio], 0, data[i].length);
            }
        }

        return subsampling_feature;
    }

    /**
     * get audio feature from file
     *
     * @param audio_path : audio path
     * @return : feature
     */
    public static float[][] get_feature_from_file(String audio_path) {
        float[] audio_data = read_wav_from_file(audio_path);
        float[][] feature = Melspectrogram.melspectrogram(audio_data, 16000, 512, 160, 512, 128);
        feature = T(log(feature));
        feature = concat_frame(feature, 3, 0);
        feature = subsampling(feature, 3);
        return feature;
    }


    public static float[][][] context_mask(int seq_len, int left, int right) {
        float[][][] mask = new float[seq_len][seq_len][1];
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                if (i - left <= j && j <= i + right) {
                    mask[i][j][0] = 0;
                } else {
                    mask[i][j][0] = 1;
                }
            }
        }
        return mask;
    }


    public static float[] flatten(float[][][] data) {
        int dim_1 = data.length;
        int dim_2 = data[0].length;
        int dim_3 = data[0][0].length;
        float[] temp = new float[dim_1 * dim_2 * dim_3];
        for (int i = 0; i < dim_1; i++) {
            for (int j = 0; j < dim_2; j++) {
                for (int k = 0; k < dim_3; k++) {
                    temp[i * dim_2 * dim_3 + j * dim_3 + k] = data[i][j][k];
                }
            }
        }
        return temp;
    }

    public static float[][] get_feature(short[] audio) {
        int audioLength = audio.length;
        float[] audio_data = new float[audioLength];
        for (int i = 0; i < audioLength; i++) {
            audio_data[i] = audio[i];
        }
        float[][] feature = Melspectrogram.melspectrogram(audio_data, 16000, 512, 160, 512, 128);
        feature = T(log(feature));
        feature = concat_frame(feature, 3, 0);
        feature = subsampling(feature, 3);
        return feature;
    }

    public static float[][] get_feature2(String audio_path) {
        float[] audio_data = read_wav_from_file(audio_path);
        float[][] feature = Melspectrogram.melspectrogram(audio_data, 16000, 2048, 160, 320, 128);
        feature = Melspectrogram.amplitude_to_db(feature);
        feature = T(feature);
        return feature;
    }

    public static void main(String[] args) {
        float[] a = read_wav_from_file("src/5_1812_20170628135834.wav");
        for (float v : a) {
            System.out.print(v+"\t");
        }
        System.out.println((a.length));

//        float[][] feature = get_feature2("src/5_1812_20170628135834.wav");
//        System.out.println(Arrays.toString(feature[300]));
    }
}
