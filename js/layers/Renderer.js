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
  
  scenery.Renderer = {
    Canvas: {},
    DOM: {},
    SVG: {},
    WebGL: {}
  };
  var Renderer = scenery.Renderer;
  
  // set these later so we have self references
  Renderer.Canvas.defaultLayerType = new scenery.LayerType( scenery.CanvasLayer, 'canvas', scenery.Renderer.Canvas, {
    // default arguments here
  } );
  Renderer.DOM.defaultLayerType = new scenery.LayerType( scenery.DOMLayer, 'dom', scenery.Renderer.DOM, {
    // default arguments here
  } );
  Renderer.SVG.defaultLayerType = new scenery.LayerType( scenery.SVGLayer, 'svg', scenery.Renderer.SVG, {
    // default arguments here
  } );
  
  // add shortcuts for the default layer types
  scenery.CanvasDefaultLayerType = Renderer.Canvas.defaultLayerType;
  scenery.DOMDefaultLayerType = Renderer.DOM.defaultLayerType;
  scenery.SVGDefaultLayerType = Renderer.SVG.defaultLayerType;
  
  // and shortcuts so we can index in with shorthands like 'svg', 'dom', etc.
  Renderer.canvas = Renderer.Canvas;
  Renderer.dom = Renderer.DOM;
  Renderer.svg = Renderer.SVG;
  Renderer.webgl = Renderer.WebGL;
  
  return Renderer;
} );
