// Copyright 2002-2014, University of Colorado Boulder

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
  var WebGLLayer = require( 'SCENERY/layers/WebGLLayer' );
  var Util = require( 'SCENERY/util/Util' );

  // BORROWED from Mr Doob (mrdoob.com), then borrowed from Pixi.js
  var hasWebGLSupport = Util.isWebGLSupported();

  scenery.Renderer = function Renderer( layerConstructor, name, bitmask, defaultOptions ) {
    this.layerConstructor = layerConstructor;
    this.name = name;
    this.bitmask = bitmask;
    this.defaultOptions = defaultOptions;

    this.defaultLayerType = this.createLayerType( {} ); // default options are handled in createLayerType
  };
  var Renderer = scenery.Renderer;

  Renderer.prototype = {
    constructor: Renderer,

    createLayerType: function( rendererOptions ) {
      return new scenery.LayerType( this.layerConstructor, this.name, this.bitmask, this, _.extend( {}, this.defaultOptions, rendererOptions ) );
    }
  };

  Renderer.Canvas = new Renderer( scenery.CanvasLayer, 'canvas', scenery.bitmaskSupportsCanvas, {} );
  Renderer.DOM = new Renderer( scenery.DOMLayer, 'dom', scenery.bitmaskSupportsDOM, {} );
  Renderer.SVG = new Renderer( scenery.SVGLayer, 'svg', scenery.bitmaskSupportsSVG, {} );
  if ( hasWebGLSupport ) {
    Renderer.WebGL = new Renderer( scenery.WebGLLayer, 'webgl', scenery.bitmaskSupportsWebGL, {} );
  }

  // add shortcuts for the default layer types
  scenery.CanvasDefaultLayerType = Renderer.Canvas.defaultLayerType;
  scenery.DOMDefaultLayerType = Renderer.DOM.defaultLayerType;
  scenery.SVGDefaultLayerType = Renderer.SVG.defaultLayerType;
  if ( hasWebGLSupport ) {
    scenery.WebGLDefaultLayerType = Renderer.WebGL.defaultLayerType;
  }

  // and shortcuts so we can index in with shorthands like 'svg', 'dom', etc.
  Renderer.canvas = Renderer.Canvas;
  Renderer.dom = Renderer.DOM;
  Renderer.svg = Renderer.SVG;
  if ( hasWebGLSupport ) {
    Renderer.webgl = Renderer.WebGL;
  }

  return Renderer;
} );
