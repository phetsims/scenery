// Copyright 2002-2013, University of Colorado

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
  
  /*---------------------------------------------------------------------------*
  * OLD Renderer handling (deprecated)
  *----------------------------------------------------------------------------*/
  
  // cached defaults
  var defaults = {};
  
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
  
  // add shortcuts for the default layer types
  scenery.CanvasDefaultLayerType = Renderer.Canvas.defaultLayerType;
  scenery.DOMDefaultLayerType    = Renderer.DOM.defaultLayerType;
  scenery.SVGDefaultLayerType    = Renderer.SVG.defaultLayerType;
  
  // and shortcuts so we can index in with shorthands like 'svg', 'dom', etc.
  Renderer.canvas = Renderer.Canvas;
  Renderer.dom = Renderer.DOM;
  Renderer.svg = Renderer.SVG;
  Renderer.webgl = Renderer.WebGL;
  
  
  /*---------------------------------------------------------------------------*
  * Renderer bitmask flags
  *----------------------------------------------------------------------------*/
  
  // ensure that these bitmasks weren't changed in scenery.js
  assert && assert( scenery.bitmaskSupportsCanvas === 0x0000001 );
  assert && assert( scenery.bitmaskSupportsSVG    === 0x0000002 );
  assert && assert( scenery.bitmaskSupportsDOM    === 0x0000004 );
  assert && assert( scenery.bitmaskSupportsWebGL  === 0x0000008 );
  
  // one renderer is required
  Renderer.bitmaskCanvas                  = scenery.bitmaskSupportsCanvas; // 0x0000001
  Renderer.bitmaskSVG                     = scenery.bitmaskSupportsSVG;    // 0x0000002
  Renderer.bitmaskDOM                     = scenery.bitmaskSupportsDOM;    // 0x0000004
  Renderer.bitmaskWebGL                   = scenery.bitmaskSupportsWebGL;  // 0x0000008
  // 10, 20, 40, 80 reserved for future renderers
  
  // fitting group (2 bits)
  Renderer.bitmaskFitting                 = 0x0000300; // bitmask that covers all of the states
  Renderer.bitmaskFitFullScene            = 0x0000000; // fit the full scene (invalid in transformed/single-cached situations (how would that work?), required for boundsless objects)
  Renderer.bitmaskFitLoose                = 0x0000100; // for now, round out to something like 32 or 128 pixel increments?
  Renderer.bitmaskFitTight                = 0x0000200; // tight fit, updates whenever it is changed
  Renderer.bitmaskFitHybrid               = 0x0000300; // custom minimization strategy
  
  // general options
  Renderer.bitmaskForceAcceleration       = 0x0000400; // 
  Renderer.bitmaskSkipBounds              = 0x0000800; // forces full scene fitting for SVG/Canvas unless there is a guaranteed bounds, so don't use in transformed/single-cached situations unless there is a bounds guarantee
  
  // canvas options
  Renderer.bitmaskCanvasLowResolution     = 0x0001000;
  Renderer.bitmaskCanvasNoPruning         = 0x0002000;
  Renderer.bitmaskCanvasNoDirtyBounds     = 0x0004000;
  Renderer.bitmaskCanvasBeforeAfterBounds = 0x0008000;
  
  // SVG optimizations group (2 bits)
  Renderer.bitmaskSVGOptimizations        = 0x0030000
  Renderer.bitmaskSVGOptimizeAuto         = 0x0000000; // auto for text-rendering/shape-rendering/image-rendering
  Renderer.bitmaskSVGOptimizeSpeed        = 0x0010000; // optimizeSpeed for text-rendering/shape-rendering/image-rendering
  Renderer.bitmaskSVGOptimizeQuality      = 0x0020000; // optimizeQuality for shape-rendering/image-rendering, geometricPrecision for text-rendering
  Renderer.bitmaskSVGOptimizeCrisp        = 0x0030000; // optimizeQuality for image-rendering, crispEdges for shape-rendering, optimizeLegibility for text-rendering
  
  // svg options
  Renderer.bitmaskSVGCollapse             = 0x0040000;
  
  Renderer.isCanvas = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskCanvas ) !== 0;
  };
  Renderer.isSVG = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskSVG ) !== 0;
  };
  Renderer.isDOM = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskDOM ) !== 0;
  };
  Renderer.isWebGL = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskWebGL ) !== 0;
  };
  Renderer.getFitStrategy = function( bitmask ) {
    return Renderer.fitStrategies[bitmask & Renderer.bitmaskFitting];
  };
  Renderer.isAccelerationForced = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskForceAcceleration ) !== 0;
  };
  Renderer.isSkipBounds = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskSkipBounds ) !== 0;
  };
  Renderer.isCanvasLowResolution = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskCanvasLowResolution ) !== 0;
  };
  Renderer.isCanvasNoPruning = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskCanvasNoPruning ) !== 0;
  };
  Renderer.isCanvasNoDirtyBounds = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskCanvasNoDirtyBounds ) !== 0;
  };
  Renderer.isCanvasBeforeAfterBounds = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskCanvasBeforeAfterBounds ) !== 0;
  };
  Renderer.getSVGOptimizations = function( bitmask ) {
    return Renderer.svgOptimizations[bitmask & Renderer.bitmaskSVGOptimizations];
  };
  
  
  /*---------------------------------------------------------------------------*
  * Fit strategies
  *----------------------------------------------------------------------------*/
  
  // TODO: fill out fit strategies, determine parameters
  Renderer.fitStrategies = {};
  Renderer.fitStrategies.bitmaskFitFullScene = function() {};
  Renderer.fitStrategies.bitmaskFitLoose = function() {};
  Renderer.fitStrategies.bitmaskFitTight = function() {};
  Renderer.fitStrategies.bitmaskFitHybrid = function() {};
  
  
  /*---------------------------------------------------------------------------*
  * SVG quality settings
  *----------------------------------------------------------------------------*/
  
  // SVG qualities for text-rendering, shape-rendering, image-rendering
  Renderer.svgOptimizations = {};
  Renderer.svgOptimizations.bitmaskSVGOptimizeAuto = {
    text: 'auto',
    shape: 'auto',
    image: 'auto'
  };
  Renderer.svgOptimizations.bitmaskSVGOptimizeSpeed = {
    text: 'optimizeSpeed',
    shape: 'optimizeSpeed',
    image: 'optimizeSpeed'
  };
  Renderer.svgOptimizations.bitmaskSVGOptimizeQuality = {
    text: 'geometricPrecision',
    shape: 'optimizeQuality',
    image: 'optimizeQuality'
  };
  Renderer.svgOptimizations.bitmaskSVGOptimizeCrisp = {
    text: 'optimizeLegibility',
    shape: 'crispEdges',
    image: 'optimizeQuality'
  };
  
  return Renderer;
} );
