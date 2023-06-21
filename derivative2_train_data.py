#!/usr/bin/python

'''
  Purpose: Build derivative training data by overlaying negative samples over positive samples
   Positive Samples - Hail and Rain
   Negative Samples - Other
'''
#! TODO: add white noise and overlay
#! Lookat denoising autoencoders

from __future__ import print_function

import csv, datetime
import os, sys, glob, random
import numpy as np
import tensorflow as tf
import logging
import pydub

flags = tf.compat.v1.flags
flags.DEFINE_string('in_dir', None, 'Path to the input samples directory')
flags.DEFINE_integer('num_derivatives', 5, 'Number of derivative files to generate by overlaying other files')
flags.DEFINE_integer('limit', 0, 'for debug 0 is unset')
flags.DEFINE_float('reduce_by_dB', 15.0, 'Reduce the negative samples by dB before overlaying')
flags.DEFINE_string('out_dir', None, 'Path to the output samples directory')
flags.DEFINE_string('out_csv_file', None, 'Output CSV file with derivative transformation info')
flags.DEFINE_string('log_file', './derivative_train_data.log', 'Path to the output log file')
FLAGS = flags.FLAGS

def init_and_get_logger():
    logger = logging.getLogger('')
    if not logger.handlers:
      logging.basicConfig(filename=FLAGS.log_file, filemode='w', format='%(levelname)s: %(message)s', level=logging.DEBUG)
      console = logging.StreamHandler()
      console.setLevel(logging.WARNING)
      console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
      logger.addHandler(console)
    return logger

def generate_derivative_files(logger, input_file, neg_files, DERIVATIVE_COUNT, quiet_by_dB, output_dir, derivative_tf_info):
    logger.info("Generating training samples from {0:s}".format(input_file))
    orig_sound = pydub.AudioSegment.from_file(input_file, format="wav")
    orig_sound = pydub.effects.normalize(orig_sound)
    file_idx = 1
    prefix_output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0])
    # randomize following
    silent_cnt = random.randint(1, DERIVATIVE_COUNT-1)
    neg_filelist = random.sample(population=neg_files, k=DERIVATIVE_COUNT-silent_cnt)
    audio_len  = len(orig_sound)  # Audio len in ms
    audio_half = int(0.5*audio_len)

    silence = pydub.AudioSegment.silent(duration=1000, frame_rate=16000)
    ## print (len(silence.get_array_of_samples()))

    for i in range(silent_cnt) : # Add no more than 50% of silence
      marker1 = random.randint(0, audio_half)
      marker2 = marker1 + random.randint(audio_half, audio_len)
      marker2 = audio_len if marker2 > audio_len else marker2

      new_sound = silence[:marker1] + orig_sound[marker1:marker2] + silence[marker2:audio_len]
      new_sound = new_sound.set_sample_width(2)
      ## print (len(new_sound.get_array_of_samples()))
      new_sound_file = prefix_output_file + "_silent_" + str(file_idx) + ".wav"
      new_sound.export(new_sound_file, format="wav")

      derivative_tf_info.append([input_file, marker1, marker2, new_sound_file])
      file_idx = file_idx + 1

    for neg_file in neg_filelist:
      marker1 = random.randint(0, audio_half)
      marker2 = marker1 + random.randint(audio_half, audio_len)
      marker2 = audio_len if marker2 > audio_len else marker2
      new_sound_file = prefix_output_file + "_insert_" + str(file_idx) + ".wav"
      logger.info(" Inserting " + neg_file)

      neg_file_sound = pydub.AudioSegment.from_file(neg_file, format="wav")
      neg_file_sound = pydub.effects.normalize(neg_file_sound)
      neg_file_sound = neg_file_sound - quiet_by_dB

      new_sound = neg_file_sound[:marker1] + orig_sound[marker1:marker2] + neg_file_sound[marker2:audio_len]
      new_sound = new_sound + random.randint(-5,5) 
      new_sound = new_sound.set_sample_width(2)
      ## print (len(new_sound.get_array_of_samples()))
      new_sound.export(new_sound_file, format="wav")

      derivative_tf_info.append([input_file, neg_file, new_sound_file])
      file_idx = file_idx + 1

    return

def write_csv_output(csv_output_file, derivative_tf_info):
    with open(csv_output_file, 'wb') as fp:
      fp.write("# Date Created {0:s}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
      fp.write("# Original File, Negative File, Derivative File\n")
      wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
      for tf_item in derivative_tf_info:
        wr.writerow(tf_item)

def main(_):
    logger = init_and_get_logger()
    if FLAGS.reduce_by_dB and FLAGS.reduce_by_dB == 0:
      logger.error("Invalid dB value 0.0 specified")
      return
    quiet_by_dB = 0.0
    if FLAGS.reduce_by_dB:
      quiet_by_dB = FLAGS.reduce_by_dB
    if FLAGS.in_dir and os.path.isdir(FLAGS.in_dir) == False:
      logger.error("input directory {0:s} is not found".format(FLAGS.in_dir))
      return
    if FLAGS.out_dir and os.path.isdir(FLAGS.out_dir) == False:
      logger.error("output directory {0:s} is not found".format(FLAGS.out_dir))
      return
    if FLAGS.in_dir:
      INPUT_DIR = FLAGS.in_dir
    else:
      INPUT_DIR = '/home/vagrawal/hail_project/final_data'
    if FLAGS.out_dir:
      OUTPUT_DIR = FLAGS.out_dir
    else:
      OUTPUT_DIR = INPUT_DIR
    if FLAGS.out_csv_file:
      out_csv_file = FLAGS.out_csv_file
    else:
      out_csv_file = "derivative_tf_info" + "_" + str(quiet_by_dB) + "dB" + ".csv"
    limit = 0
    if FLAGS.limit:
      limit = FLAGS.limit

    INPUT_DIR = os.path.expanduser(INPUT_DIR)
    OUTPUT_DIR = os.path.expanduser(OUTPUT_DIR)
    DERIVATIVE_COUNT = FLAGS.num_derivatives
    CSV_OUTPUT = OUTPUT_DIR + "/" + out_csv_file

    logger.info("Input directory {0:s}".format(INPUT_DIR))
    logger.info("Output directory {0:s}".format(OUTPUT_DIR))
    logger.info("Number of derivatives of a file {0:d}".format(DERIVATIVE_COUNT))
    logger.info("CSV with derivative transform details {0:s} ".format(CSV_OUTPUT))

    pos_types = ['rain', 'hail']
    neg_types = ['other_train', 'other_engine']
    pos_files, neg_files = [], []
    for file_type in pos_types:
      for file_path in glob.glob(INPUT_DIR + '/*' + file_type + '*train/*.wav'):
        pos_files.append(file_path)
    for file_type in neg_types:
      for file_path in glob.glob(INPUT_DIR + '/' + file_type + '/*.wav'):
        neg_files.append(file_path)

    num_neg_files = len(neg_files)
    if num_neg_files == 0:
      logger.error("Not enough -ve files", num_neg_files)
      
    if DERIVATIVE_COUNT > num_neg_files:
      DERIVATIVE_COUNT = num_neg_files

    if limit != 0 :
      pos_files = pos_files[:limit]

    derivative_tf_info = []
    logger = init_and_get_logger()
    for file_path in pos_files:
      file_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(file_path)))
      new_file_dir = file_dir + '_derived2'
      if not os.path.exists(new_file_dir):
        os.makedirs(new_file_dir)
      generate_derivative_files(logger, file_path, neg_files, DERIVATIVE_COUNT, quiet_by_dB, new_file_dir, derivative_tf_info)

    write_csv_output(CSV_OUTPUT, derivative_tf_info)

if __name__ == '__main__':
  tf.compat.v1.run()
