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
  
  // for now, use the basic Layer constructors. consider adding options for more advanced use in the future
  Renderer.Canvas.createLayerType = function( rendererOptions ) {
    return new scenery.LayerType( scenery.CanvasLayer, 'canvas', scenery.Renderer.Canvas, _.extend( {
      // default arguments here
    }, rendererOptions ) );
  };
  Renderer.DOM.createLayerType = function( rendererOptions ) {
    return new scenery.LayerType( scenery.DOMLayer, 'dom', scenery.Renderer.DOM, _.extend( {
      // default arguments here
    }, rendererOptions ) );
  };
  Renderer.SVG.createLayerType = function( rendererOptions ) {
    return new scenery.LayerType( scenery.SVGLayer, 'svg', scenery.Renderer.SVG, _.extend( {
      // default arguments here
    }, rendererOptions ) );
  };
  
  // add shortcuts for the default layer types
  scenery.CanvasDefaultLayerType = Renderer.Canvas.defaultLayerType = Renderer.Canvas.createLayerType( {} );
  scenery.DOMDefaultLayerType    = Renderer.DOM.defaultLayerType    = Renderer.DOM.createLayerType( {} );
  scenery.SVGDefaultLayerType    = Renderer.SVG.defaultLayerType    = Renderer.SVG.createLayerType( {} );
  
  // and shortcuts so we can index in with shorthands like 'svg', 'dom', etc.
  Renderer.canvas = Renderer.Canvas;
  Renderer.dom = Renderer.DOM;
  Renderer.svg = Renderer.SVG;
  Renderer.webgl = Renderer.WebGL;
  
  return Renderer;
} );
