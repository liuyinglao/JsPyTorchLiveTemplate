/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import * as React from 'react';
import {
  Camera,
  Canvas,
  CanvasRenderingContext2D,
  ImageUtil,
  media,
  MobileModel,
  Module,
  Tensor,
  torch,
  torchvision,
} from 'react-native-pytorch-core';
import {
  ActivityIndicator,
  Button,
  Image,
  StyleSheet,
  TouchableHighlight,
  View,
  Text,
  Dimensions,
} from 'react-native';
const MODEL =
  'https://github.com/pytorch/live/releases/download/v0.1.0/deeplabv3.ptl';

// images are selected from the PASCAL VOC2012 dataset
const images = [
  'https://raw.githubusercontent.com/liuyinglao/TestData/main/airplane_resized.jpeg',
  'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/02.jpg',
  'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/06.jpg',
  'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/13.jpg',
  'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/07.jpg',
  'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/08.jpg',
  'https://github.com/pytorch/hub/raw/master/images/deeplab1.png',
];

// colors are selected from https://reactnative.dev/docs/colors#color-keywords
// they are corresponding to the following class:
// ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
//  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
//  'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
const palette = [
  'aliceblue',
  'cornflowerblue',
  'cornsilk',
  'crimson',
  'cyan',
  'darkblue',
  'darkcyan',
  'darkgoldenrod',
  'darkgray',
  'antiquewhite',
  'aqua',
  'aquamarine',
  'beige',
  'blue',
  'blueviolet',
  'brown',
  'burlywood',
  'cadetblue',
  'chartreuse',
  'chocolate',
  'coral',
];

export default function Playground() {
  const [isProcessing, setIsProcessing] = React.useState(false);
  const context2DRef = React.useRef(null);
  const [isReady, setIsReady] = React.useState(false);
  const modelRef = React.useRef(null);

  const [imageIndex, setImageIndex] = React.useState(0);
  const [scaleFactor, setScaleFactor] = React.useState(1);
  const img = React.useMemo(() => images[imageIndex], [imageIndex]);

  React.useEffect(() => {

    async function loadModel() {
      const startLoadModelTime = global.performance.now();
      const modelPath = await MobileModel.download(MODEL);
      modelRef.current = await torch.jit._loadForMobile(modelPath);
      console.log('model load time: ', global.performance.now() - startLoadModelTime);
      setIsReady(true);
    }
    loadModel();
    return () => {
      modelRef.current = undefined;
      setIsReady(false);
    };
  }, []);

  const handleImage = React.useCallback(async (imageUrl) => {
    const ctx = context2DRef.current;
    if (ctx !== null) {
      ctx.clear();
      await ctx.invalidate();
    }
    let model = modelRef.current;
    if (model == null) {
      const modelPath = await MobileModel.download(MODEL);
      model = await torch.jit._loadForMobile(modelPath);
    }
    setIsProcessing(true);

    const startPackTime = global.performance.now();

    const image = await ImageUtil.fromURL(imageUrl);
    const height = image.getHeight();
    const width = image.getWidth();
    const blob = media.toBlob(image);
    let tensor = torch.fromBlob(blob, [
      height,
      width,
      3,
    ]);
    tensor = tensor.to({dtype: torch.float32});
    tensor = tensor.div(255);
    tensor = tensor.permute([2, 0, 1]);

    let normalize = torchvision.transforms.normalize(
      [0.485, 0.456, 0.406],
      [0.229, 0.224, 0.225],
    );
    tensor = normalize(tensor);
    tensor = tensor.unsqueeze(0);

    console.log('model pack time: ', global.performance.now() - startPackTime);
    const startInferenceTime = global.performance.now();

    const segmented = await model.forward(tensor);

    console.log('model inference time: ', global.performance.now() - startInferenceTime);
    const startUnpackTime = global.performance.now();

    const t = segmented.out.squeeze(0).argmax({dim:0});

    console.log('model unpack time: ', global.performance.now() - startUnpackTime);

    if (ctx !== null) {
        const windowWidth = Dimensions.get('window').width;

        const newScaleFactor = Math.min(250 / height, windowWidth / width);

        // hardcoding the scale factor to fit most of the example images for demo
        ctx.scale(newScaleFactor / scaleFactor, newScaleFactor / scaleFactor);
        setScaleFactor(newScaleFactor);
        const startDrawImageTime = global.performance.now();
        ctx.drawImage(image, 0, 0, width, height);
        const colors = t.data;

        const startDrawMaskTime = global.performance.now();

        for (let i = 0; i < height; i++) {
          for (let j = 0; j < width; j++) {
            let cls = colors[i*width+j];
            if (cls > 0) {
              ctx.fillStyle=palette[cls];
              ctx.fillCircle(j, i, 1);
            }
          }
        }
        console.log("image draw time: ", startDrawMaskTime - startDrawImageTime);
        console.log("mask draw time: ", global.performance.now() - startDrawMaskTime);
      }
      await ctx.invalidate();


    setIsProcessing(false);
  }, [setIsProcessing, setScaleFactor, scaleFactor]);

  function updateImageIndex(diff) {
    setImageIndex(v => {
      const nextVal = v + diff;
      if (nextVal < 0) {
        return v;
      }
      else if (nextVal >= images.length) {
        return v;
      }
      return nextVal;
    });
  }

  return (
    <View style={styles.container}>
      <View style={styles.navigationButtonContainer}>
        <Button
          style={styles.navigationButton}
          title="Prev"
          onPress={() => {
            updateImageIndex(-1);
          }}
        />
        <Button
          style={styles.navigationButton}
          title="Next"
          onPress={() => {
            updateImageIndex(1);
          }}
        />
      </View>
      <Image style={styles.image} source={{uri: images[imageIndex]}} />
      {isReady ? <Button
        title="Segment pictures"
        onPress={() => handleImage(images[imageIndex])}
      /> :  <ActivityIndicator color="white" size="large" />
      }
      {isProcessing && <ActivityIndicator color="white" size="large" />}
      <View style={styles.canvas}>
        <Canvas
          style={StyleSheet.absoluteFill}
          onContext2D={ctx => {
            context2DRef.current = ctx;
          }}
        />
      </View>
    </View>
  );
}



const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  navigationButtonContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
  },
  navigationButton: {
    margin: 20,
  },
  canvas: {
    marginVertical: 20,
    width: '100%',
    height: 250,
  },
  image: {
    width: '100%',
    height: 300,
    resizeMode: 'cover',
  }
});
