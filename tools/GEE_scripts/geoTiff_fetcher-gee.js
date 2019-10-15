/* Script to be run in Google Earth Engine code editor
 * url: https://code.earthengine.google.com/
 *
 */


var loc = {
  // USDA - NAIP & NASS_CDL
  name: "FresnoCDL-T2V",
  period: ['2016-01-01', '2016-12-31'],
  area: 
    ee.Geometry.Rectangle({
    coords:[[-119.65,37.05],
            [-120.25,36.45]],
    geodesic: false
    })
};

////////////////// NAIP Images Geotiff ////////////////////////////
var naip_dataset = ee.ImageCollection('USDA/NAIP/DOQQ')
                  .filter(ee.Filter.date(loc.period[0], loc.period[1]));
var img = naip_dataset.filterBounds(loc.area).median();

////////////////// NASS - Cropland ////////////////////////////////
var cdl_dataset = ee.ImageCollection("USDA/NASS/CDL")
                  .filter(ee.Filter.date(loc.period[0], loc.period[1]));
var cdl_img = cdl_dataset.filterBounds(loc.area).median();

img = img.addBands(cdl_img)
///////////////////////////////////////////////////////////////////



var filepath = "USDA/CDL_NASS_with_NAIP/" + loc.name + '/' + loc.name 
Export.image.toCloudStorage({
  image: img,
  description: loc.name,
  bucket: 'satellite-raw-data', 
  fileNamePrefix: filepath,
  scale: 1,
  region: loc.area,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13,
  formatOptions: { 
  cloudOptimized: true,
  // fileDimensions: 4096
  }
});
