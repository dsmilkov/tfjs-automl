import tensorflow as tf
import numpy as np

with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], '.')
  img_raw = tf.reshape(tf.io.read_file('test_image.jpg'), [1]).eval()
  tofloat, boxes, scores, classes = sess.run(
    ['ToFloat:0', 'detection_boxes:0', 'detection_scores:0', 'detection_classes:0'],
    feed_dict={'encoded_image_string_tensor:0': img_raw})
  print('tofloat min/max/shape', np.min(tofloat), np.max(tofloat), tofloat.shape)
  print('boxes min/max, shape', np.min(boxes), np.max(boxes), boxes.shape)
  print('scores min/max, shape', np.min(scores), np.max(scores), scores.shape)
  print('classes min/max, shape', np.min(classes), np.max(classes), classes.shape)
  
  print('~~~~~~~~~~ SCORES ~~~~~~~~~~~~~~')
  print(scores)

  print('~~~~~~~~~~ BOXES ~~~~~~~~~~~~~~')
  print(boxes)

  print('~~~~~~~~~~ CLASSES ~~~~~~~~~~~~~~')
  print(classes)
  # res = sess.run('index_to_string/hash_table:0')
  # print(res.dtype, res.shape)
  # res.astype('bytes')
  # print(res)
