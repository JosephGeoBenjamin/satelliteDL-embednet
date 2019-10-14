/* Script to be run in Google Earth Engine code editor
 * url: https://code.earthengine.google.com/
 *
 */


var location = {
  name: "Fresno-T2V",
  area: 
    ee.Geometry.Rectangle({
    coords:[[-119.65,37.05],
            [-120.25,36.45]],
    geodesic: false
    })
};


var dataset = ee.ImageCollection('USDA/NAIP/DOQQ')
                  .filter(ee.Filter.date('2016-01-01', '2018-12-31'));


var img = dataset.filterBounds(location.area).median();

var filepath = "Naip-USA-dumps/cities/" + location.name + '/' + location.name 
Export.image.toCloudStorage({
  image: img,
  description: location.name,
  bucket: 'satellite-raw-data', 
  fileNamePrefix: filepath,
  scale: 1,
  region: location.area,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13,
  formatOptions: { 
  cloudOptimized: true,
  fileDimensions: 4096
  }
});
