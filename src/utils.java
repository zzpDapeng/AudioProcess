import java.util.Arrays;

public class utils {
    public static void show_2d_array(float[][] array, int head_n){
        int num = 0;
        for (float[] row:array){
            System.out.println(Arrays.toString(row));
            if (++num >= head_n){
                break;
            }
        }
    }
    public static void show_2d_array(float[][] array){
        int head_n = array.length;
        show_2d_array(array, head_n);
    }

}
