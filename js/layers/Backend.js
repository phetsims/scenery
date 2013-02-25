// Copyright 2002-2012, University of Colorado

/**
 * An enumeration of different back-end technologies used for rendering. It also essentially
 * represents the API that nodes need to implement to be used with this specified back-end.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  require( 'SCENERY/layers/LayerType' );
  require( 'SCENERY/layers/CanvasLayer' );
  require( 'SCENERY/layers/DOMLayer' );
  require( 'SCENERY/layers/SVGLayer' );
  
  // cached defaults
  var defaults = {};
  
  scenery.Backend = {
    Canvas: {},
    DOM: {},
    SVG: {},
    WebGL: {}
  };
  var Backend = scenery.Backend;
  
  // set these later so we have self references
  Backend.Canvas.defaultLayerType = new scenery.LayerType( scenery.CanvasLayer, 'canvas', scenery.Backend.Canvas, {
    // default arguments here
  } );
  Backend.DOM.defaultLayerType = new scenery.LayerType( scenery.DOMLayer, 'dom', scenery.Backend.DOM, {
    // default arguments here
  } );
  Backend.SVG.defaultLayerType = new scenery.LayerType( scenery.SVGLayer, 'canvas', scenery.Backend.SVG, {
    // default arguments here
  } );
  
  // add shortcuts for the default layer types
  scenery.CanvasDefaultLayerType = Backend.Canvas.defaultLayerType;
  scenery.DOMDefaultLayerType = Backend.DOM.defaultLayerType;
  scenery.SVGDefaultLayerType = Backend.SVG.defaultLayerType;
  
  return Backend;
} );
