# python loss_reporter.py $HOME/working_dir/cnn_dm/pointer-original/scheduled-sampling-e2e/ 62830

import sys
from glob import glob
import tensorflow as tf
import time

log_root = sys.argv[1]
#filter_iter = int(sys.argv[2])

while True:
  for rouge_based_eval in [False, True]:
      event_files = sorted(glob('{}/eval_{}/events*'.format(log_root, "rouge" if rouge_based_eval else "loss")))
      best_loss = 0
      best_rouge = 0
      best_rouge_step = 0
      best_step = 0
      for ef in event_files:
        try:
          for e in tf.train.summary_iterator(ef):
            for v in e.summary.value:
              step = e.step
              if 'running_avg_loss/decay' in v.tag:
                running_avg_loss = v.simple_value
                if best_loss == 0 or running_avg_loss < best_loss:
                  best_loss = running_avg_loss
                  best_step = step
              if 'running_greedy_rouge' in v.tag:
                running_greedy_rouge = v.simple_value
                if best_rouge == 0 or running_greedy_rouge > best_rouge:
                  best_rouge = running_greedy_rouge
                  best_rouge_step = step
        except Exception as ex:
          print(ex.message)
          print("nothing to show for {}".format('{}'.format(ef)))
          continue
      if rouge_based_eval:
        best_loss = best_rouge
        best_step = best_rouge_step
      print(
      'resotring best {} from the current logs: {}\tstep: {}'.format("rouge" if rouge_based_eval else "loss",
                                                                     best_loss,
                                                                     best_step))
  print("--------------------------------------------------------")
  time.sleep(15)