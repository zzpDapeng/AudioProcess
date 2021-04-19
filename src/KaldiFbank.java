import java.math.BigInteger;

/**
 * Kaldi Fbank Feature Extraction
 */
public class KaldiFbank {
    private static final float MILLISECONDS_TO_SECONDS = 0.001f;
    private static final double EPSILON = 1.1920928955078125e-07;


    public static float[][] fbank(float[] waveform, int sample_frequency,
                                  int num_mel_bins, int frame_length, int frame_shift) {
        // _get_waveform_and_window_properties()
        int window_shift = (int) (sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS);
        int window_size = (int) (sample_frequency * frame_length * MILLISECONDS_TO_SECONDS);
        int padded_window_size = _next_power_of_2(window_size);  //round_to_power_of_two:True

        // _get_window(), strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
        // signal_log_energy is not used. So use simplified _get_window(), which only returns strided_input.
        String window_type = "povey";
        float blackman_coeff = 0.42f;
        float preemphasis_coefficient = 0.97f;
        float[][] strided_input = _get_window(waveform, padded_window_size, window_size, window_shift,
                window_type, blackman_coeff, preemphasis_coefficient);
        int m = strided_input.length;

        // (m, padded_window_size // 2 + 1)
        float[][] power_spectrum = rfft(strided_input);

        float low_freq = 20.0f;
        float high_freq = 0.0f;
        // size (num_mel_bins, padded_window_size // 2)
        float[][] mel_energies = get_mel_banks(num_mel_bins, padded_window_size, sample_frequency,
                low_freq, high_freq);
        mel_energies = pad(mel_energies, new int[]{0, 1}, "constant", 0);

        //mel_energies = (power_spectrum * mel_energies).sum(dim=2)
        float[][] bank_feature = new float[m][num_mel_bins];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < num_mel_bins; j++) {
                for (int k = 0; k < padded_window_size / 2; k++) {
                    bank_feature[i][j] += power_spectrum[i][k] * mel_energies[j][k];
                }
                bank_feature[i][j] = (float) Math.log(Math.max(bank_feature[i][j], EPSILON));
            }
        }
        return bank_feature;
    }

    private static float[][] _get_window(float[] waveform, int padded_window_size, int window_size,
                                         int window_shift, String window_type, float blackman_coeff,
                                         float preemphasis_coefficient) {
        float[][] strided_input = _get_strided(waveform, window_size, window_shift);  //snip_edges:True
        int m = strided_input.length;
        int n = strided_input[0].length;
        //  remove_dc_offset
        float[] row_means = mean(strided_input, 1);

        // strided_input = strided_input - row_means
        float[][] offset_strided_input = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                strided_input[i][j] -= row_means[i];
                offset_strided_input[i][j] = j == 0 ? strided_input[i][j] : strided_input[i][j - 1];
            }
        }
        // row_energy,Compute the log energy of each row/frame before applying preemphasis and window function
//        float[] signal_log_energy = _get_log_energy(strided_input, energy_floor);  // size (m), signal_log_energy is not used.
        float[] window_function = _feature_window_function(window_type, window_size, blackman_coeff);  // size (window_size)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                strided_input[i][j] -= preemphasis_coefficient * offset_strided_input[i][j];
                // Apply window_function to each row/frame
                strided_input[i][j] *= window_function[j];
            }
        }

        if (padded_window_size != window_size) {
            int padding_right = padded_window_size - window_size;
            strided_input = pad(strided_input, new int[]{0, padding_right}, "constant", 0);
        }
        return strided_input;
    }

    private static float[][] get_mel_banks(int num_bins, int window_length_padded, int sample_freq,
                                           float low_freq, float high_freq) {
        int num_fft_bins = window_length_padded / 2;
        float nyquist = 0.5f * sample_freq;
        if (high_freq <= 0.0) {
            high_freq += nyquist;
        }
        float fft_bin_width = sample_freq / (float) window_length_padded;
        float mel_low_freq = mel_scale_scalar(low_freq);
        float mel_high_freq = mel_scale_scalar(high_freq);

        float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1);

//        float[] center_freqs = inverse_mel_scale(center_mel);  // not used.
        float[] fft_bins = new float[num_fft_bins];
        for (int i = 0; i < num_fft_bins; i++) {
            fft_bins[i] = fft_bin_width * i;
        }
        float[] mel = mel_scale(fft_bins);

        float left_mel;
        float center_mel;
        float right_mel;
        float[][] bins = new float[num_bins][num_fft_bins];
        for (int i = 0; i < num_bins; i++) {
            left_mel = mel_low_freq + i * mel_freq_delta;
            center_mel = mel_low_freq + (i + 1) * mel_freq_delta;
            right_mel = mel_low_freq + (i + 2) * mel_freq_delta;
            for (int j = 0; j < num_fft_bins; j++) {
                bins[i][j] = (float) Math.max(0.0, Math.min((mel[j] - left_mel) / (center_mel - left_mel),
                        (right_mel - mel[j]) / (right_mel - center_mel)));
            }
        }
        return bins;
    }

    /**
     * Compute mel scale of tensor(float[])
     *
     * @param freq: float[]
     * @return res: float[]
     */
    private static float[] mel_scale(float[] freq) {
        float[] res = new float[freq.length];
        for (int i = 0; i < freq.length; i++) {
            res[i] = mel_scale_scalar(freq[i]);
        }
        return res;
    }

    /**
     * not used for now, which occurs in get_mel_banks.
     *
     * @param mel_freq:
     * @return res:
     */
    private static float[] inverse_mel_scale(float[] mel_freq) {
        float[] res = new float[mel_freq.length];
        for (int i = 0; i < mel_freq.length; i++) {
            res[i] = (float) (700 * (Math.pow(Math.E, mel_freq[i] / 1127.0f) - 1.0f));
        }
        return res;
    }

    private static float mel_scale_scalar(float freq) {
        return (float) (1127.0f * Math.log(1.0f + freq / 700.0f));
    }

    private static float[][] rfft(float[][] strided_input) {
        int m = strided_input.length;
        int padded_window_size = strided_input[0].length;
        int n = padded_window_size / 2 + 1;
        float[][] power_spectrum = new float[m][n];
        for (int i = 0; i < m; i++) {
            float[][] fft = FFT.rfft(strided_input[i], padded_window_size);
            for (int j = 0; j < n; j++) {
                power_spectrum[i][j] = fft[0][j] * fft[0][j] + fft[1][j] * fft[1][j];
            }
        }
        return power_spectrum;
    }

    private static float[] _feature_window_function(String window_type, int window_size, double blackman_coeff) {
        float[] window_function = new float[window_size];
        float div = window_size - 1;
        if (window_type.equals("povey")) {
            for (int i = 0; i < window_size; i++) {
//                window_function[i] = (float) Math.pow(Math.sin((Math.PI * i) / div), 2 * 0.85);
                window_function[i] = (float) Math.pow(0.5 * (1 - Math.cos(2 * Math.PI * i / div)), 0.85);
            }
        } else {
            throw new IllegalArgumentException("Unsupported window type:" + window_type + ". Support povey mode only.");
        }
        return window_function;
    }

    /**
     * pytorch torch.nn.functional.pad, 2d input.
     *
     * @param input：
     * @param p1d:  for example: (1,0), not support multi dim for now.
     * @param mode: only support 'replicate' mode for now.
     * @return padded_input:
     */
    private static float[][] pad(float[][] input, int[] p1d, String mode, int value) {
        int left = p1d[0];
        int right = p1d[1];
        int row = input.length;
        int col = input[0].length;
        float[][] padded_input;
        if (mode.equals("replicate")) {
            int padded_col = col + left + right;
            padded_input = new float[row][padded_col];
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < left; j++) {
                    padded_input[i][j] = input[i][0];
                }
                System.arraycopy(input[i], 0, padded_input[i], left, col);
                for (int j = left + col; j < padded_col; j++) {
                    padded_input[i][j] = input[i][col - 1];
                }
            }
        } else if (mode.equals("constant")) {
            int padded_col = col + left + right;
            padded_input = new float[row][padded_col];
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < left; j++) {
                    padded_input[i][j] = value;
                }
                System.arraycopy(input[i], 0, padded_input[i], left, col);
                for (int j = left + col; j < padded_col; j++) {
                    padded_input[i][j] = value;
                }
            }
        } else {
            throw new IllegalArgumentException("Unsupported pad mode:" + mode);
        }
        return padded_input;
    }

    private static float[][] pad(float[][] input, int[] p1d, String mode) {
        return pad(input, p1d, mode, 0);
    }

    private static float[] _get_log_energy(float[][] strided_input, float energy_floor) {
        int m = strided_input.length;
        int n = strided_input[0].length;
        float[] log_energy = new float[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                log_energy[i] += Math.max(Math.pow(strided_input[i][j], 2), KaldiFbank.EPSILON);
            }
            log_energy[i] = (float) Math.log(log_energy[i]);
            log_energy[i] = Math.max(log_energy[i], energy_floor);
        }
        return log_energy;
    }

    private static float[] mean(float[][] strided_input, int dim) {
        int col = strided_input.length;
        int row = strided_input[0].length;
        float[] mean_out;
        if (dim == 1) {
            mean_out = new float[col];
            for (int i = 0; i < col; i++) {
                for (int j = 0; j < row; j++) {
                    mean_out[i] += strided_input[i][j] / (float) row;
                }
            }
        } else {
            mean_out = new float[row];
            for (int i = 0; i < row; i++) {
                for (float[] floats : strided_input) {
                    mean_out[i] += floats[i] / (float) col;
                }
            }
        }
        return mean_out;
    }

    private static float[][] _get_strided(float[] waveform, int window_size, int window_shift) {
        int num_samples = waveform.length;
        int[] strides = new int[]{window_shift, 1};
        //snip_edges:True
        int m;
        if (num_samples < window_size) {
            return new float[0][0];  //torch.empty((0, 0))
        } else {
            m = 1 + (num_samples - window_size) / window_shift;
        }
        int[] size = new int[]{m, window_size};
        return _as_strided(waveform, size, strides);
    }

    private static float[][] _as_strided(float[] waveform, int[] size, int[] strides) {
        float[][] strided = new float[size[0]][size[1]];
        for (int i = 0; i < size[0]; i++) {
            int start_index = strides[0] * i;
            for (int j = 0; j < size[1]; j++) {
                strided[i][j] = waveform[start_index + j * strides[1]];
            }
        }
        return strided;
    }

    private static int _next_power_of_2(int x) {
        if (x == 0) {
            return 1;
        } else {
            return (int) Math.pow(2, BitLength(x - 1));
        }
    }

    private static int BitLength(int x) {
        BigInteger b = new BigInteger(String.valueOf(x));
        return b.bitLength();
    }

    public static float[][] fbank(float[] waveform) {
        int sample_frequency = 16000;
        int num_mel_bins = 80;
        int frame_length = 25;
        int frame_shift = 10;
        return fbank(waveform, sample_frequency, num_mel_bins, frame_length, frame_shift);
    }

    public static void main(String[] args) {
        float[] waveform = AudioProcess.read_wav_from_file("src/5_1812_20170628135834.wav");
        float[][] feature = fbank(waveform);
        utils.show_2d_array(feature);
    }
}
