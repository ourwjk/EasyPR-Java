package org.easypr.core;

import static org.bytedeco.javacpp.opencv_core.CV_32FC1;
import static org.easypr.core.CoreFunc.features;

import java.util.HashMap;
import java.util.Map;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_ml.CvANN_MLP;
import org.easypr.util.Convert;

/**
 * @author Created by fanwenjie
 * @author lin.yao.hello
 * 
 */
public class CharsIdentify {

    public CharsIdentify() {
        loadModel();

        if (this.map.isEmpty()) {
            map.put("zh_cuan", "宸�");
            map.put("zh_e", "閯�");
            map.put("zh_gan", "璧�");
            map.put("zh_gan1", "鐢�");
            map.put("zh_gui", "璐�");
            map.put("zh_gui1", "妗�");
            map.put("zh_hei", "榛�");
            map.put("zh_hu", "娌�");
            map.put("zh_ji", "鍐�");
            map.put("zh_jin", "娲�");
            map.put("zh_jing", "浜�");
            map.put("zh_jl", "鍚�");
            map.put("zh_liao", "杈�");
            map.put("zh_lu", "椴�");
            map.put("zh_meng", "钂�");
            map.put("zh_min", "闂�");
            map.put("zh_ning", "瀹�");
            map.put("zh_qing", "闈�");
            map.put("zh_qiong", "鐞�");
            map.put("zh_shan", "闄�");
            map.put("zh_su", "鑻�");
            map.put("zh_sx", "鏅�");
            map.put("zh_wan", "鐨�");
            map.put("zh_xiang", "婀�");
            map.put("zh_xin", "鏂�");
            map.put("zh_yu", "璞�");
            map.put("zh_yu1", "娓�");
            map.put("zh_yue", "绮�");
            map.put("zh_yun", "浜�");
            map.put("zh_zang", "钘�");
            map.put("zh_zhe", "娴�");
        }
    }

    /**
     * @param input
     * @param isChinese
     * @return
     */
    public String charsIdentify(final Mat input, final Boolean isChinese, final Boolean isSpeci) {
        String result = "";

        Mat f = features(input, this.predictSize);

        int index = classify(f, isChinese, isSpeci);

        if (!isChinese) {
            result = String.valueOf(strCharacters[index]);
        } else {
            String s = strChinese[index - numCharacter];
            result = map.get(s);
        }
        return result;
    }

    private int classify(final Mat f, final Boolean isChinses, final Boolean isSpeci) {
        int result = -1;
        Mat output = new Mat(1, numAll, CV_32FC1);

        ann.predict(f, output);

        int ann_min = (!isChinses) ? ((isSpeci) ? 10 : 0) : numCharacter;
        int ann_max = (!isChinses) ? numCharacter : numAll;

        float maxVal = -2;

        for (int j = ann_min; j < ann_max; j++) {
            float val = Convert.toFloat(output.ptr(0, j));

            if (val > maxVal) {
                maxVal = val;
                result = j;
            }
        }

        return result;
    }

    private void loadModel() {
        loadModel(this.path);
    }

    public void loadModel(String s) {
        this.ann.clear();
        this.ann.load(s, "ann");
    }

    static boolean hasPrint = false;

    public final void setModelPath(String path) {
        this.path = path;
    }

    public final String getModelPath() {
        return this.path;
    }

    private CvANN_MLP ann = new CvANN_MLP();

    private String path = "res/model/ann.xml";

    private int predictSize = 10;

    private Map<String, String> map = new HashMap<String, String>();

    private final char strCharacters[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
            'F', 'G', 'H', /* 娌℃湁I */'J', 'K', 'L', 'M', 'N', /* 娌℃湁O */'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
            'Z' };
    private final static int numCharacter = 34; // 娌℃湁I鍜�0,10涓暟瀛椾笌24涓嫳鏂囧瓧绗︿箣鍜�

    private final String strChinese[] = { "zh_cuan" /* 宸� */, "zh_e" /* 閯� */, "zh_gan" /* 璧� */, "zh_gan1"/* 鐢� */,
            "zh_gui"/* 璐� */, "zh_gui1"/* 妗� */, "zh_hei" /* 榛� */, "zh_hu" /* 娌� */, "zh_ji" /* 鍐� */, "zh_jin" /* 娲� */,
            "zh_jing" /* 浜� */, "zh_jl" /* 鍚� */, "zh_liao" /* 杈� */, "zh_lu" /* 椴� */, "zh_meng" /* 钂� */,
            "zh_min" /* 闂� */, "zh_ning" /* 瀹� */, "zh_qing" /* 闈� */, "zh_qiong" /* 鐞� */, "zh_shan" /* 闄� */,
            "zh_su" /* 鑻� */, "zh_sx" /* 鏅� */, "zh_wan" /* 鐨� */, "zh_xiang" /* 婀� */, "zh_xin" /* 鏂� */, "zh_yu" /* 璞� */,
            "zh_yu1" /* 娓� */, "zh_yue" /* 绮� */, "zh_yun" /* 浜� */, "zh_zang" /* 钘� */, "zh_zhe" /* 娴� */};
    @SuppressWarnings("unused")
    private final static int numChinese = 31;

    private final static int numAll = 65; /* 34+31=65 */
}
