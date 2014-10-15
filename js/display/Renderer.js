// Copyright 2002-2014, University of Colorado Boulder


/**
 * An enumeration of different back-end technologies used for rendering. It also essentially represents the API that
 * nodes need to implement to be used with this specified back-end.
 *
 * We use a bitmask to represent renderers currently, in a way that can be logically-ANDed in order to obtain
 * information about "what renderer can support all of these Nodes?"
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  // now it's a namespace
  scenery.Renderer = {};
  var Renderer = scenery.Renderer;

  //OHTWO TODO: rename to take advantage of lack of deprecated names? (remove bitmask prefix)

  /*---------------------------------------------------------------------------*
   * Renderer bitmask flags
   *----------------------------------------------------------------------------*/

  // ensure that these bitmasks weren't changed in scenery.js
  assert && assert( scenery.bitmaskSupportsCanvas === 0x0000001 );
  assert && assert( scenery.bitmaskSupportsSVG === 0x0000002 );
  assert && assert( scenery.bitmaskSupportsDOM === 0x0000004 );
  assert && assert( scenery.bitmaskSupportsWebGL === 0x0000008 );

  // these will need to be updated if another renderer option is given (modify order bitmasks below also)
  Renderer.bitmaskRendererArea = scenery.bitmaskRendererArea;   // 0x000000F
  // one renderer is required
  Renderer.bitmaskCanvas = scenery.bitmaskSupportsCanvas; // 0x0000001
  Renderer.bitmaskSVG = scenery.bitmaskSupportsSVG;       // 0x0000002
  Renderer.bitmaskDOM = scenery.bitmaskSupportsDOM;       // 0x0000004
  Renderer.bitmaskWebGL = scenery.bitmaskSupportsWebGL;   // 0x0000008
  // 10, 20, 40, 80 reserved for future renderers

  // fitting group (2 bits)
  Renderer.bitmaskFitting = 0x0000300;      // bitmask that covers all of the states
  Renderer.bitmaskFitFullScene = 0x0000000; // fit the full scene (invalid in transformed/single-cached situations (how would that work?), required for boundsless objects)
  Renderer.bitmaskFitLoose = 0x0000100;     // for now, round out to something like 32 or 128 pixel increments?
  Renderer.bitmaskFitTight = 0x0000200;     // tight fit, updates whenever it is changed
  Renderer.bitmaskFitHybrid = 0x0000300;    // custom minimization strategy

  // general options
  Renderer.bitmaskForceAcceleration = 0x0000400;
  Renderer.bitmaskSkipBounds = 0x0000800; // forces full scene fitting for SVG/Canvas unless there is a guaranteed bounds, so don't use in transformed/single-cached situations unless there is a bounds guarantee

  // canvas options
  Renderer.bitmaskCanvasLowResolution = 0x0001000;
  Renderer.bitmaskCanvasNoPruning = 0x0002000;
  Renderer.bitmaskCanvasNoDirtyBounds = 0x0004000;
  Renderer.bitmaskCanvasBeforeAfterBounds = 0x0008000;

  // SVG optimizations group (2 bits)
  Renderer.bitmaskSVGOptimizations = 0x0030000;
  Renderer.bitmaskSVGOptimizeAuto = 0x0000000;    // auto for text-rendering/shape-rendering/image-rendering
  Renderer.bitmaskSVGOptimizeSpeed = 0x0010000;   // optimizeSpeed for text-rendering/shape-rendering/image-rendering
  Renderer.bitmaskSVGOptimizeQuality = 0x0020000; // optimizeQuality for shape-rendering/image-rendering, geometricPrecision for text-rendering
  Renderer.bitmaskSVGOptimizeCrisp = 0x0030000;   // optimizeQuality for image-rendering, crispEdges for shape-rendering, optimizeLegibility for text-rendering

  // svg options
  Renderer.bitmaskSVGCollapse = 0x0040000;

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

  // returns the part of the bitmask that should contain only Canvas/SVG/DOM/WebGL flags
  //OHTWO TODO: use this instead of direct access to bitmaskRendererArea
  Renderer.getStrippedBitmask = function( bitmask ) {
    return bitmask & Renderer.bitmaskRendererArea;
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

  Renderer.createOrderBitmask = function( firstRenderer, secondRenderer, thirdRenderer, fourthRenderer ) {
    firstRenderer = firstRenderer || 0;
    secondRenderer = secondRenderer || 0;
    thirdRenderer = thirdRenderer || 0;
    fourthRenderer = fourthRenderer || 0;

    return firstRenderer |
           ( secondRenderer << 4 ) |
           ( thirdRenderer << 8 ) |
           ( fourthRenderer << 12 );
  };
  Renderer.bitmaskOrderFirst = function( bitmask ) {
    return bitmask & 0x000000F;
  };
  Renderer.bitmaskOrderSecond = function( bitmask ) {
    return ( bitmask >> 4 ) & 0x000000F;
  };
  Renderer.bitmaskOrderThird = function( bitmask ) {
    return ( bitmask >> 8 ) & 0x000000F;
  };
  Renderer.bitmaskOrderFourth = function( bitmask ) {
    return ( bitmask >> 12 ) & 0x000000F;
  };
  Renderer.pushOrderBitmask = function( bitmask, renderer ) {
    assert && assert( typeof bitmask === 'number' );
    assert && assert( typeof renderer === 'number' );
    var rendererToInsert = renderer;
    for ( var i = 0; i < 20; i += 4 ) {
      var currentRenderer = ( bitmask >> i ) & 0x000000F;
      if ( currentRenderer === rendererToInsert ) {
        return bitmask;
      }
      else if ( currentRenderer === 0 ) {
        // place the renderer and exit
        bitmask = bitmask | ( rendererToInsert << i );
        return bitmask;
      }
      else {
        // clear out that slot
        bitmask = ( bitmask & ~( 0x000000F << i ) );

        // place in the renderer to insert
        bitmask = bitmask | ( rendererToInsert << i );

        rendererToInsert = currentRenderer;
      }

      // don't walk over and re-place our initial renderer
      if ( rendererToInsert === renderer ) {
        return bitmask;
      }
    }

    throw new Error( 'pushOrderBitmask overflow' );
  };

  Renderer.createSelfDrawable = function( instance, node, selfRenderer ) {
    if ( Renderer.isCanvas( selfRenderer ) ) {
      return node.createCanvasDrawable( selfRenderer, instance );
    }
    else if ( Renderer.isSVG( selfRenderer ) ) {
      return node.createSVGDrawable( selfRenderer, instance );
    }
    else if ( Renderer.isDOM( selfRenderer ) ) {
      return node.createDOMDrawable( selfRenderer, instance );
    }
    else {
      // assert so that it doesn't compile down to a throw (we want this function to be optimized)
      assert && assert( 'Unrecognized renderer, maybe we don\'t support WebGL yet?: ' + selfRenderer );
    }
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
