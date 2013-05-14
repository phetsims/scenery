// Copyright 2002-2012, University of Colorado

/**
 * An enumeration of different back-end technologies used for rendering. It also essentially
 * represents the API that nodes need to implement to be used with this specified back-end.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  require( 'SCENERY/layers/LayerType' );
  require( 'SCENERY/layers/CanvasLayer' );
  require( 'SCENERY/layers/DOMLayer' );
  require( 'SCENERY/layers/SVGLayer' );
  
  // cached defaults
  var defaults = {};
  
  scenery.Renderer = function( layerConstructor, name, defaultOptions ) {
    this.layerConstructor = layerConstructor;
    this.name = name;
    this.defaultOptions = defaultOptions;
    
    this.defaultLayerType = this.createLayerType( {} ); // default options are handled in createLayerType
  };
  var Renderer = scenery.Renderer;
  
  Renderer.prototype = {
    constructor: Renderer,
    
    createLayerType: function( rendererOptions ) {
      return new scenery.LayerType( this.layerConstructor, this.name, this, _.extend( {}, this.defaultOptions, rendererOptions ) );
    }
  };
  
  Renderer.Canvas = new Renderer( scenery.CanvasLayer, 'canvas', {} );
  Renderer.DOM = new Renderer( scenery.DOMLayer, 'dom', {} );
  Renderer.SVG = new Renderer( scenery.SVGLayer, 'svg', {} );
  
  // add shortcuts for the default layer types
  scenery.CanvasDefaultLayerType = Renderer.Canvas.defaultLayerType;
  scenery.DOMDefaultLayerType    = Renderer.DOM.defaultLayerType;
  scenery.SVGDefaultLayerType    = Renderer.SVG.defaultLayerType;
  
  // and shortcuts so we can index in with shorthands like 'svg', 'dom', etc.
  Renderer.canvas = Renderer.Canvas;
  Renderer.dom = Renderer.DOM;
  Renderer.svg = Renderer.SVG;
  Renderer.webgl = Renderer.WebGL;
  
  return Renderer;
} );
