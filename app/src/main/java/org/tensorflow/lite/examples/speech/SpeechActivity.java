t rem/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Demonstrates how to run an audio recognition model in Android.

This example loads a simple speech recognition model trained by the tutorial at
https://www.tensorflow.org/tutorials/audio_training

The model files should be downloaded automatically from the TensorFlow website,
but if you have a custom model you can update the LABEL_FILENAME and
MODEL_FILENAME constants to point to your own files.

The example application displays a list view with all of the known audio labels,
and highlights each one when it thinks it has detected one through the
microphone. The averaging of results to give a more reliable signal happens in
the RecognizeCommands helper class.
*/

package org.tensorflow.lite.examples.speech;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import androidx.appcompat.widget.SwitchCompat;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
//import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import com.google.android.material.bottomsheet.BottomSheetBehavior;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.util.concurrent.LinkedBlockingQueue;



/**
 * An activity that listens for audio and then uses a TensorFlow model to detect particular classes,
 * by default a small set of action words.
 */
public class SpeechActivity extends Activity
    implements View.OnClickListener, CompoundButton.OnCheckedChangeListener {

  // Constants that control the behavior of the recognition code and model
  // settings. See the audio recognition tutorial for a detailed explanation of
  // all these, but you should customize them to match your training settings if
  // you are running your own model.
  private static final int SAMPLE_RATE = 22050;

  private static final String HANDLE_THREAD_NAME = "Background";
  private static Map<String,int[]> phone_files;
  private static Map<String,String> text_files;

  private final int stream_len=32;
  private final int overlap = 8;


  // UI elements.
  private static final String LOG_TAG = SpeechActivity.class.getSimpleName();

  private File logDir;


  private final ReentrantLock tfLiteLock = new ReentrantLock();
  private final ReentrantLock benchmarkLock = new ReentrantLock();
//  private final ReentrantLock inferLock = new ReentrantLock();
  private final ReentrantLock melganLock = new ReentrantLock();
  private final ReentrantLock decodeLock = new ReentrantLock();
  private Condition ready = benchmarkLock.newCondition();

  private NnApiDelegate nnApiDelegate = null;
  private GpuDelegate gpuDelegate = null;

  private LinearLayout bottomSheetLayout;
  private LinearLayout gestureLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  private final Interpreter.Options tfLiteOptions = new Interpreter.Options();

  private MappedByteBuffer tfLiteModel_encoder;
  private MappedByteBuffer tfLiteModel_decoder;
  private MappedByteBuffer tfMelGan;
  private Interpreter tfLite_encoder;
  private Interpreter tfLite_decoder;
  private Interpreter melgantfLite;
  private ImageView bottomSheetArrowImageView;

  private TextView sentence;
  private Button playButton;
  private Button benchmarkButton;
  private Button loadModelButton;
  private Spinner model_list;
  private Spinner text_list;
//  private CheckBox checkBox;
  private TextView fastspeechTextView, inferenceTimeTextView,rtfTextView,firstchunkTextView,durationView,rtfpFrameView;
  private ImageView plusImageView, minusImageView;
  private SwitchCompat apiSwitchCompat;
  private Switch useGPU;
  private TextView threadsTextView;
  private long lastProcessingTimeMs;
  private Handler handler = new Handler();
  private TextView selectedTextView = null;
  private HandlerThread backgroundThread;
  private Handler backgroundHandler;
  private Thread synthesisThread,encoder,decoder,melgan,playaudio;
  private final LinkedBlockingQueue<short[]> fragmentQueue = new LinkedBlockingQueue<>();


  private final AudioTrack audio = new AudioTrack(AudioManager.STREAM_MUSIC,
          SAMPLE_RATE, //sample rate
          AudioFormat.CHANNEL_OUT_MONO, //1 channel
          AudioFormat.ENCODING_PCM_16BIT, // 16-bit
          4096,
          AudioTrack.MODE_STREAM );

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  @SuppressLint("WrongViewCast")
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    // Set up the UI.

    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_sc_activity_speech);


    firstchunkTextView = findViewById(R.id.fist_chunk_info);
    inferenceTimeTextView = findViewById(R.id.infer_info);
    rtfTextView = findViewById(R.id.rtf_info);
    fastspeechTextView = findViewById(R.id.fastspeech_info);
    durationView = findViewById(R.id.duration);
    rtfpFrameView = findViewById(R.id.rtfperframe_info);

    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    gestureLayout = findViewById(R.id.gesture_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);

    threadsTextView = findViewById(R.id.threads);
    plusImageView = findViewById(R.id.plus);
    minusImageView = findViewById(R.id.minus);
    apiSwitchCompat = findViewById(R.id.api_info_switch);

    useGPU = findViewById(R.id.gpuswitch);
    useGPU.setText("OFF");
    playButton = findViewById(R.id.button);
    benchmarkButton = findViewById(R.id.benchmark);
    loadModelButton = findViewById(R.id.load_model);
    apiSwitchCompat.setOnCheckedChangeListener(this);

    sentence = findViewById(R.id.textView2);
    text_list = findViewById(R.id.spinner);
    model_list = findViewById(R.id.spinner2);
    /* create a list of items for the spinner. */

    phone_files = new HashMap<>();
    text_files = new HashMap<>();
    new Thread(loadResourceRunner).start();
    new Thread(loadTextRunner).start();

    text_list.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {

      @Override
      public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
        sentence.setText(text_files.get(text_list.getSelectedItem()));
      }

      @Override
      public void onNothingSelected(AdapterView<?> parent) {
        sentence.setText("");
      }
    });

//set the spinners adapter to the previously created one.
    String[] models = new String[6];
    models[0] = "fastspeech2plusv3.melganv3";
    models[1] = "fastspeech2plusv3.melganv4";
    models[2] = "fastspeech2plusv3.melganv3_dq";
    models[3] = "fastspeech2plusv3.melganv4_dq";
    models[4] = "fastspeech2plusv3.melganv3_fl16";
    models[5] = "fastspeech2plusv3.melganv4_fl16";

    ArrayAdapter<String> adapterModel = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, models);
    model_list.setAdapter(adapterModel);

    benchmarkButton.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v)
      {
        try {
          benchmark();
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
      }
    });

    playButton.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v)
      {
        int[] tempInputBuffer = phone_files.get(text_list.getSelectedItem());
        try {
          synthesis(tempInputBuffer,false, "");
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
      }
    });

    useGPU.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
      public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        // do something, the isChecked will be
        // true if the switch is in the On position
        if(isChecked){
          Log.v(LOG_TAG,"use GPU delegate");
          recreateInterpreter();
          useGPU.setText("GPU");
        } else {
          useGPU.setText("OFF");
        }
      }
    });

    loadModelButton.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v)
      {
        long startTime = new Date().getTime();
        String model_name = (String) model_list.getSelectedItem();
        try {
          tfLiteModel_encoder = loadModelFile(getAssets(), model_name+"/fastspeech_encoder.tflite");
          tfLiteModel_decoder = loadModelFile(getAssets(), model_name+"/fastspeech_decoder.tflite");
          tfMelGan = loadModelFile(getAssets(), model_name+"/melgan_v3_32.tflite");
          recreateInterpreter();
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
        recreateInterpreter();
        long lastProcessingTimeMs1 = new Date().getTime() - startTime;
        Log.v(LOG_TAG, "load fastspeech and melgan: " + lastProcessingTimeMs1);
      }
    });



    ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
        new ViewTreeObserver.OnGlobalLayoutListener() {
          @Override
          public void onGlobalLayout() {
            gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
            int height = gestureLayout.getMeasuredHeight();

            sheetBehavior.setPeekHeight(height);
          }
        });
    sheetBehavior.setHideable(false);

    sheetBehavior.setBottomSheetCallback(
        new BottomSheetBehavior.BottomSheetCallback() {
          @Override
          public void onStateChanged(@NonNull View bottomSheet, int newState) {
            switch (newState) {
              case BottomSheetBehavior.STATE_HIDDEN:
                break;
              case BottomSheetBehavior.STATE_EXPANDED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                }
                break;
              case BottomSheetBehavior.STATE_COLLAPSED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                }
                break;
              case BottomSheetBehavior.STATE_DRAGGING:
                break;
              case BottomSheetBehavior.STATE_SETTLING:
                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                break;
            }
          }

          @Override
          public void onSlide(@NonNull View bottomSheet, float slideOffset) {}
        });

    plusImageView.setOnClickListener(this);
    minusImageView.setOnClickListener(this);

    try {
      tfLiteModel_encoder = loadModelFile(getAssets(), "fastspeech2plusv3.melganv3/fastspeech_encoder.tflite");
      tfLiteModel_decoder = loadModelFile(getAssets(), "fastspeech2plusv3.melganv3/fastspeech_decoder.tflite");
      tfMelGan = loadModelFile(getAssets(), "fastspeech2plusv3.melganv3/melgan_v3_32.tflite");
      recreateInterpreter();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

  }


  private static short[] normPCM(float[] pcm) {
    short[] shortNormedPCM = new short[pcm.length];

    float maxAbsoluteValue = Float.NEGATIVE_INFINITY;
    for (float step : pcm) {
      maxAbsoluteValue = Math.max(Math.abs(step), maxAbsoluteValue);
    }

    double ratio = Math.min(Short.MAX_VALUE,maxAbsoluteValue) / maxAbsoluteValue;
    for (int i = 0; i < pcm.length; ++i) {
      shortNormedPCM[i] = (short) (pcm[i]*Short.MAX_VALUE*ratio);
    }
    return shortNormedPCM;
}

  private void benchmark() throws InterruptedException {
    SimpleDateFormat formatter = new SimpleDateFormat("dd-MM-yyyy_HH:mm:ss");
    logDir = new File(getApplicationContext().getFilesDir(),
            "benchmark_" + formatter.format(new Date()));
    if (!logDir.mkdir()) {
      throw new RuntimeException("Cannot create Directory " + logDir.getAbsolutePath());
    }
    try {
      //BufferedWriter for performance, true to set append to file flag
      BufferedWriter buf = new BufferedWriter(new FileWriter(logDir+"/log.file", true));
      String text = "filename,firstChunk_time,fastspeech,minRTFperframe,maxRTFperframe,avg_rtf,rtf,totalInferencetime";
      buf.append(text);
      buf.newLine();
      buf.close();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    AtomicInteger count = new AtomicInteger();
    int total = phone_files.size();
    new Thread(()-> {
      benchmarkLock.lock();
      for (Map.Entry<String, int[]> entry : phone_files.entrySet()) {
        String key = entry.getKey();
        int[] value = entry.getValue();
        try {
          ready = benchmarkLock.newCondition();
          synthesis(value, true, key);
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
        count.getAndIncrement();
        final int done = count.get();
        runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    sentence.setText("" + done + "/" + total);
                  }
                }
        );
        try {
          ready.await();
        } catch (InterruptedException e) {
          e.printStackTrace();
        }

      }
      benchmarkLock.unlock();
    }).start();
  }

  private void synthesis(int[] tempInputBuffer,boolean log, String filename) throws InterruptedException {
    if (playaudio!=null){
      playaudio.interrupt();
      playaudio.join();
      playaudio=null;
    }

    if (synthesisThread!=null){
      synthesisThread.interrupt();
      synthesisThread.join();
      synthesisThread=null;
    }
    runOnUiThread(
            new Runnable() {
              @Override
              public void run() {
                inferenceTimeTextView.setText("0 ms");
                fastspeechTextView.setText("0 ms");
                firstchunkTextView.setText("0 ms");
                rtfTextView.setText("0");
                rtfpFrameView.setText("0");
                durationView.setText("0 ms");
              }
            }
    );

    int len_inp = tempInputBuffer.length;
    int[] inputBuffer = new int[300];
    int[] seq_pos = new int[300];
    for (int i = 0; i < 300; ++i) {
      if (i<len_inp){
        inputBuffer[i] = tempInputBuffer[i];
        seq_pos[i]=i;
      } else{
        seq_pos[i]=0;
        inputBuffer[i] = 0;
      }
    }
    if(!log) {
      playaudiosteam();
    }
    synthesisThread = new Thread(() -> {
      benchmarkLock.lock();
      Log.v(LOG_TAG, "Synthesis");

//    prepare input for encoder
      Object[] inputMap = new Object[3];
      inputMap[0]=inputBuffer;
      inputMap[1]=seq_pos;
      inputMap[2]=len_inp;
//    prepare output for encoder
      HashMap<Integer, Object> outputEncoder = new HashMap<>();
      float[][][] outputEncode = new float[1][300][256];
      int[][] duration = new int[1][300];
//      int[] total_len = new int[1];
      outputEncoder.put(0,outputEncode);
      outputEncoder.put(1,duration);
//      outputEncoder.put(2,total_len);

      long fastspeech_proc_time;
      long encoderFastspeech_time;
      long firstChunk_time = 0;
      float minRTFperframe = 100;
      float maxRTFperframe = 0;
      float duration_pred;
      float rtf;
      float avg_rtf;


      // Run the fastspeech2 encoder.
      long startTime = new Date().getTime();
      tfLiteLock.lock();
      run_encoder(inputMap,outputEncoder);

//    prepare pinput decoder
      float[][][] output_am = new float[1][len_inp][256];
      int [][] pre_duration = new int[1][len_inp];

      float[][][] tflite_enc_out = (float[][][]) outputEncoder.get(0);
      int [][] tflite_duration_out = (int[][]) outputEncoder.get(1);
      tfLiteLock.lock();
      int temp_frame_len = 0;
      for (int i = 0; i < len_inp; ++i) {
        output_am[0][i] = tflite_enc_out[0][i];
        pre_duration[0][i] = tflite_duration_out[0][i];
        temp_frame_len+=pre_duration[0][i];
      }
      final int frame_len = temp_frame_len;
      tfLiteLock.unlock();
      Object[] inputDecode = new Object[2];
      inputDecode[0]=output_am;
      inputDecode[1]=pre_duration;

      tfLite_decoder.resizeInput(0,new int[]{1,len_inp,256});
      tfLite_decoder.resizeInput(1,new int[]{1,len_inp});
//      tfLite_decoder.allocateTensors();
//    prepare output decoder
      float[][][] outputDecode = new float[1][frame_len][80];
      HashMap<Integer, Object> outputFastspeech = new HashMap<>();
      outputFastspeech.put(0, outputDecode);


//     run fastspeech2 decoder
      decodeLock.lock();
      try {
        run_decoder(inputDecode,outputFastspeech);
//        float [] pad = new float[80];
//        Arrays.fill(pad, 0);
//        float[][][] temp_mel = new float[1][1000][80];
//        for (int i = 0; i < 1000; ++i) {
//          if (i<566){
//            temp_mel[0][i]= outputDecode[0][i];
//          } else {
//            temp_mel[0][i]=pad;
//          }
//        }
//        Object[] inputMel = {temp_mel};
      } finally {
        fastspeech_proc_time = new Date().getTime()-startTime;
        duration_pred = (float) (frame_len*256.0/22050*1000);
        if(!log) {
          runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      fastspeechTextView.setText(fastspeech_proc_time + " ms");
                      durationView.setText(frame_len * 256.0 / 22050 * 1000 + " ms");
                    }
                  }
          );
        }
      }

//    run melgan
      int temp_overlap=0;
      int start = 0;
      int end;
//    prepare output melgan
//      melgantfLite.resizeInput(0,new int[]{1,stream_len,80});
//      float[][][] outputWav = new float[1][frame_len*256][1];
      float[][][] outputWav = new float[1][stream_len*256][1];
      HashMap<Integer, Object> outputWave = new HashMap<>();
      outputWave.put(0, outputWav);


      melganLock.lock();
      try{
        int melgancount = 0;
        float tt_rtf_per_frame = 0;
        float rtf_per_frame = 0;

        while(start<frame_len) {
          end = start + stream_len;
          if (end > frame_len) {
            temp_overlap = end - frame_len;
            start = frame_len - stream_len;
            end = frame_len;
          }

          Object[] inputMel = new Object[1];
          float[][][] temp_input_mel = new float[1][stream_len][80];
          int count = 0;
          for(int i=start;i<end;++i){
            temp_input_mel[0][count]= outputDecode[0][i];
            count++;
          }
          inputMel[0] = temp_input_mel;
          long melgan_start_infer= new Date().getTime();
          run_melgan(inputMel, outputWave);
          long melgan_process_time = new Date().getTime()-melgan_start_infer;
          if (log){
            if (start==0) {
              firstChunk_time = new Date().getTime() - startTime;
              long finalFirstChunk_time = firstChunk_time;
            } else if (end == frame_len) {
              break;
            }
          } else {
            if (start == 0) {
              firstChunk_time = new Date().getTime() - startTime;
              long finalFirstChunk_time = firstChunk_time;

              runOnUiThread(
                      new Runnable() {
                        @Override
                        public void run() {
                          firstchunkTextView.setText(finalFirstChunk_time + " ms");
                        }
                      }
              );
              queue_wave(outputWav, 0, overlap / 2);
            } else if (end == frame_len) {
              queue_wave(outputWav, temp_overlap + overlap / 2, 0);
              break;
            } else {
              queue_wave(outputWav, temp_overlap + overlap / 2, overlap / 2);
            }
          }

          rtf_per_frame = (float) (melgan_process_time/((stream_len -temp_overlap - overlap)*256.0/22050*1000));
          if (maxRTFperframe < rtf_per_frame){
            maxRTFperframe = rtf_per_frame;
          }
          if (minRTFperframe > rtf_per_frame){
            minRTFperframe = rtf_per_frame;
          }
          tt_rtf_per_frame +=rtf_per_frame;
          melgancount+=1;
          start = end - overlap;
        }
        avg_rtf = tt_rtf_per_frame/melgancount;
        if(!log) {
          runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      rtfpFrameView.setText(avg_rtf + "");
                    }
                  }
          );
        }

      } finally {
        tfLiteLock.unlock();
        decodeLock.unlock();
        melganLock.unlock();
      }

//      playaudio((float[][][]) outputWave.get(0));
//      stop audio player
      try {
        fragmentQueue.put(new short[]{});
      } catch (InterruptedException ignored) {
      }

      lastProcessingTimeMs = new Date().getTime() - startTime;
      rtf = (float) (lastProcessingTimeMs/(frame_len*256.0/22050*1000));
      runOnUiThread(
              new Runnable() {
                @Override
                public void run() {

                  Log.v(LOG_TAG, "time: " + lastProcessingTimeMs);
                  inferenceTimeTextView.setText(lastProcessingTimeMs + " ms");
                  rtfTextView.setText("" + rtf);
                }
              }
      );
      Log.v(LOG_TAG, "End synthesis");

      if (log) {
        try {
          //BufferedWriter for performance, true to set append to file flag
          BufferedWriter buf = new BufferedWriter(new FileWriter(logDir+"/log.file", true));
          String text = filename + ", " + firstChunk_time + ", " + fastspeech_proc_time + ", " + minRTFperframe + ", " + maxRTFperframe + ", " + avg_rtf + ", " + rtf + ", " + lastProcessingTimeMs;
          Log.v(LOG_TAG,"log to "+logDir+"/log.file");
          Log.v(LOG_TAG, text);
          buf.append(text);
          buf.newLine();
          buf.close();
        } catch (IOException e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
        }
      }

//      benchmarkLock.unlock();
      ready.signal();
      benchmarkLock.unlock();
    });
    synthesisThread.start();
  }

  private void run_melgan(Object[] inputMel, HashMap<Integer, Object> outputWave){
    long startTime2 = new Date().getTime();
    melgantfLite.runForMultipleInputsOutputs(inputMel, outputWave);
    long lastProcessingTimeMs2 = new Date().getTime() - startTime2;
    Log.v(LOG_TAG, "melgan time: " + lastProcessingTimeMs2);
  }

  private void run_encoder(Object[] inputMap, HashMap<Integer, Object> outputEncoder){
    long startTime1 = new Date().getTime();
    tfLite_encoder.runForMultipleInputsOutputs(inputMap, outputEncoder);
    long lastProcessingTimeMs1 = new Date().getTime() - startTime1;
    Log.v(LOG_TAG, "fastspeech encoder time: " + lastProcessingTimeMs1);
  }

  private void run_decoder(Object[] inputDecode, HashMap<Integer, Object> outputFastspeech){
    long startTime1 = new Date().getTime();
    tfLite_decoder.runForMultipleInputsOutputs(inputDecode, outputFastspeech);

    long lastProcessingTimeMs1 = new Date().getTime() - startTime1;
    Log.v(LOG_TAG, "fastspeechtime decoder: " + lastProcessingTimeMs1);
  }

  private void queue_wave(float[][][] wave, int start ,int end){
//    new Thread(()-> {
      int len = wave[0].length;
      float[] pcm = new float[len];
      for (int i = 0; i < len; ++i) {
        pcm[i] = wave[0][i][0];
      }
      short[] npcm = normPCM(pcm);
//      Log.v(LOG_TAG, "overlap:"+overlap*256+"len:"+len);
      try {
        fragmentQueue.put(Arrays.copyOfRange(npcm, start*256, len-end*256));
      } catch (InterruptedException e) {
        e.printStackTrace();
      }

//    }).start();

  }

  private void playaudiosteam(){
    playaudio = new Thread(() -> {
      audio.play();
      fragmentQueue.clear();
      Log.d(LOG_TAG, "audio player started");
      while (true) {
        try {
          if (Thread.currentThread().isInterrupted()) {
            break;
          }
          short[] fragment = fragmentQueue.take();
          if (fragment.length == 0) {
            Log.d(LOG_TAG, "audio player stopped");
            break;
          }
          audio.write(fragment, 0, fragment.length);
        } catch (InterruptedException e) {
          break;
        }
      }
    });
    playaudio.start();
  }


  @Override
  public void onClick(View v) {
    if ((v.getId() != R.id.plus) && (v.getId() != R.id.minus)) {
      return;
    }

    String threads = threadsTextView.getText().toString().trim();
    int numThreads = Integer.parseInt(threads);
    if (v.getId() == R.id.plus) {
      numThreads++;
    } else {
      if (numThreads == 1) {
        return;
      }
      numThreads--;
    }

    final int finalNumThreads = numThreads;
    threadsTextView.setText(String.valueOf(finalNumThreads));
    backgroundHandler.post(
        () -> {
          tfLiteOptions.setNumThreads(finalNumThreads);
          recreateInterpreter();
        });
  }

  @Override
  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
    backgroundHandler.post(
        () -> {
          tfLiteOptions.setUseNNAPI(isChecked);
          recreateInterpreter();
        });
    if (isChecked) apiSwitchCompat.setText("NNAPI");
    else apiSwitchCompat.setText("TFLITE");
  }

  private void recreateInterpreter() {
    tfLiteLock.lock();
    long startTime = new Date().getTime();
    try {
      if (tfLite_encoder != null) {
        tfLite_encoder.close();
        tfLite_encoder = null;
      }
      if (tfLite_decoder != null) {
        tfLite_decoder.close();
        tfLite_decoder = null;
      }
      if (melgantfLite != null) {
        melgantfLite.close();
        melgantfLite = null;
      }

      boolean useNNAPI = apiSwitchCompat.getText()=="NNAPI";
      boolean usegpu = useGPU.getText()=="GPU";
      Interpreter.Options options = new Interpreter.Options();
      CompatibilityList compatList = new CompatibilityList();
      if (useNNAPI) {
        options.setUseNNAPI(true);
      }
      if (usegpu){
      if(compatList.isDelegateSupportedOnThisDevice()){
        // if the device has a supported GPU, add the GPU delegate
        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        options.addDelegate(gpuDelegate);
        Log.v(LOG_TAG, "enable gpu delegate");
      }
      else {
        Toast.makeText(getApplicationContext(),
                "no gpu support",
                Toast.LENGTH_LONG/2)
                .show();
        usegpu=false;
      }
      }
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      options.setNumThreads(numThreads);
      melgantfLite = new Interpreter(tfMelGan, options);

      tfLiteOptions.setUseNNAPI(false);
      tfLite_encoder = new Interpreter(tfLiteModel_encoder, tfLiteOptions);

      tfLite_decoder = new Interpreter(tfLiteModel_decoder, tfLiteOptions);
      tfLite_encoder.resizeInput(0, new int[] {300});
      tfLite_encoder.allocateTensors();
      tfLite_decoder.allocateTensors();
      melgantfLite.resizeInput(0,new int[]{1,32,80});
      melgantfLite.allocateTensors();
      String model = (String) model_list.getSelectedItem();
      long time =  (new Date()).getTime() - startTime;
      String message;
      if (usegpu && useNNAPI)
        message = "Loaded " +model + " in " +time + "ms use: " + threads + " threads, use GPU and NNAPI";
      else if (usegpu) {
        message = "Loaded " + model + " in " + time + "ms use: " + threads + " threads, use GPU";
      } else {
        message = "Loaded " + model + " in " + time + "ms use: " + threads + " threads, use " + apiSwitchCompat.getText();
      }

      Toast.makeText(getApplicationContext(),
              message,
              Toast.LENGTH_LONG)
              .show();

    } finally {
      tfLiteLock.unlock();
    }
  }

  private final Runnable loadResourceRunner =
          new Runnable() {
            @Override
            public void run() {
              try {
                Scanner scanner = new Scanner(getApplicationContext().getAssets().open("benchmark_phone.txt"));
                ArrayList lines = new ArrayList<>();
                while (scanner.hasNextLine()) {
                  lines.add(scanner.nextLine());
                }
                String[] files = new String[lines.size()];

                for (int i=0; i < lines.size(); i++){
                    String[] temp = lines.get(i).toString().split("\\|");
                    temp[1] = temp[1].substring(1,temp[1].length()-1);
                    String[] phone = temp[1].split(", ");
                    int[] phoneme = new int[phone.length];
                    for (int j=0; j<phoneme.length; ++j){
                      phoneme[j] = Integer.parseInt(phone[j]);
                    }
                    phone_files.put(temp[0],phoneme);
                    files[i] = temp[0];
                }

                ArrayAdapter<String> adapter =
                        new ArrayAdapter<>(
                                getApplicationContext(), android.R.layout.simple_spinner_dropdown_item, files);
                text_list.setAdapter(adapter);
              } catch (Exception e) {
                e.printStackTrace();
                finish();
              }
            }
          };

  private final Runnable loadTextRunner =
          new Runnable() {
            @Override
            public void run() {
              try {
                Scanner scanner = new Scanner(getApplicationContext().getAssets().open("benchmark_text.txt"));
                ArrayList lines = new ArrayList<>();
                while (scanner.hasNextLine()) {
                  lines.add(scanner.nextLine());
                }
                for (int i=0; i < lines.size(); i++){
                  String[] temp = lines.get(i).toString().split("\\|");
                  text_files.put(temp[0],temp[1]);
                }
              } catch (Exception e) {
                e.printStackTrace();
                finish();
              }
            }
          };

  private void startBackgroundThread() {
    backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
    backgroundThread.start();
    backgroundHandler = new Handler(backgroundThread.getLooper());
  }

  private void stopBackgroundThread() {
    backgroundThread.quitSafely();
    try {
      backgroundThread.join();
      backgroundThread = null;
      backgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e("amlan", "Interrupted when stopping background thread", e);
    }
  }

  @Override
  protected void onResume() {
    super.onResume();

    startBackgroundThread();
  }

  @Override
  protected void onStop() {
    super.onStop();
    stopBackgroundThread();
  }
}
