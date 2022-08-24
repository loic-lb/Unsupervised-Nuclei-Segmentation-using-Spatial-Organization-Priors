import qupath.ext.stardist.StarDist2D

// Specify the model file (you will need to change this!)
var pathModel = '.../he_heavy_augment.pb'

var stardist = StarDist2D.builder(pathModel)
      .threshold(0.5)              // Prediction threshold
      .preprocess(        // Extra preprocessing steps, applied sequentially
              ImageOps.Core.subtract(100),
              ImageOps.Core.divide(100)
      )
      //.normalizePercentiles(1, 99) // Percentile normalization
      .pixelSize(1.4)              // Resolution for detection
      .build()

// Run detection for the selected objects
var imageData = getCurrentImageData()
var pathObjects = getAnnotationObjects()
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}
stardist.detectObjects(imageData, pathObjects)
println 'Done!'