const tf = require('@tensorflow/tfjs-node');
const bodyPix = require('@tensorflow-models/body-pix');
const fs = require('fs');
const Jimp = require('jimp');

class RemoveBackground {

    // BodyPix model
    _model;

    /**
     * Load body-pix model
     * @return {Promise<void>}
     * @private
     */
    async _loadModel() {
        if (!this._model) {
            const resNetModel = {
                architecture: 'ResNet50',
                outputStride: 16,
                quantBytes: 4
            };
            this._model = await bodyPix.load(resNetModel);
        }
    }

    /**
     * Make body segmentation
     * @param image - A Tensor with dtype `int32` and a 3-dimensional shape, for jpeg/png/bmp the returned Tensor shape is [height, width, channels]
     * @return {Promise<SemanticPartSegmentation>}
     * @private
     */
    async _makePrediction(image) {
        //Check if model was loaded, if not, load it
        await this._loadModel();
        return this._model.segmentPersonParts(image);
    }

    /**
     *
     * @param image - Buffer
     * @param outputFile - It's a string for the output file name
     * @return {Promise<void>}
     */
    async removeBackground(image, outputFile) {
        // Make body segmentation
        const tfImage = tf.node.decodeImage(image);
        const bodySegmentation = await this._makePrediction(tfImage);
        // Load image to jimp object
        const jimpImage = await Jimp.read(image);
        // Remove background
        let count = 0;
        // Iterate over every pixel in the image and remove the color if is not part of the body
        for (let i = 0; i < bodySegmentation.height; i++) {
            for (let j = 0; j < bodySegmentation.width; j++) {
                // -1 means that this pixel doesn't belong to a person
                if (bodySegmentation.data[count] === -1) {
                    jimpImage.setPixelColor(0x00000000, j, i);
                }
                count++;
            }
        }
        // Save the edited image
        await jimpImage.writeAsync(outputFile);
    }
}

(async () => {
    // Usage example
    const removeBg = new RemoveBackground();
    // Read image from file
    const img = fs.readFileSync('input.jpg');
    // Remove background from image
    await removeBg.removeBackground(img, 'output.png');
})();
