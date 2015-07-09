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

  Renderer.bitmaskRendererArea = 0x00000FF;
  Renderer.bitmaskCurrentRendererArea = 0x000001F;
  Renderer.bitmaskLacksOffset = 0x10000;
  Renderer.bitmaskLacksShift = 16; // number of bits between the main renderer bitmask and the "lacks" variety
  Renderer.bitmaskNodeDefault = Renderer.bitmaskRendererArea;

  Renderer.bitmaskCanvas = 0x0000001;
  Renderer.bitmaskSVG = 0x0000002;
  Renderer.bitmaskDOM = 0x0000004;
  Renderer.bitmaskWebGL = 0x0000008;
  Renderer.bitmaskPixi = 0x0000010;
  // 20, 40, 80 reserved for future renderers NOTE: update bitmaskCurrentRendererArea if they are added/removed

  // summary bits (for RendererSummary):
  Renderer.bitmaskSingleCanvas = 0x100;
  Renderer.bitmaskSingleSVG = 0x200;
  // reserved gap 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000 for future renderer-specific single information
  Renderer.bitmaskNotPainted = 0x1000;
  Renderer.bitmaskBoundsValid = 0x2000;
  // summary bits for whether a renderer could be potentially used to display a Node.
  Renderer.bitmaskLacksCanvas = Renderer.bitmaskCanvas << Renderer.bitmaskLacksShift; // 0x10000
  Renderer.bitmaskLacksSVG = Renderer.bitmaskSVG << Renderer.bitmaskLacksShift; // 0x20000
  Renderer.bitmaskLacksDOM = Renderer.bitmaskDOM << Renderer.bitmaskLacksShift; // 0x40000
  Renderer.bitmaskLacksWebGL = Renderer.bitmaskWebGL << Renderer.bitmaskLacksShift; // 0x80000
  Renderer.bitmaskLacksPixi = Renderer.bitmaskPixi << Renderer.bitmaskLacksShift; // 0x100000
  // reserved gap 0x20000, 0x40000, 0x80000 for future renderers

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
  Renderer.isPixi = function( bitmask ) {
    return ( bitmask & Renderer.bitmaskPixi ) !== 0;
  };

  var rendererMap = {
    canvas: Renderer.bitmaskCanvas,
    svg: Renderer.bitmaskSVG,
    dom: Renderer.bitmaskDOM,
    webgl: Renderer.bitmaskWebGL,
    pixi: Renderer.bitmaskPixi
  };
  Renderer.fromName = function( name ) {
    return rendererMap[ name ];
  };

  // returns the part of the bitmask that should contain only Canvas/SVG/DOM/WebGL/Pixi flags
  //OHTWO TODO: use this instead of direct access to bitmaskRendererArea
  Renderer.stripBitmask = function( bitmask ) {
    return bitmask & Renderer.bitmaskRendererArea;
  };

  Renderer.createOrderBitmask = function( firstRenderer, secondRenderer, thirdRenderer, fourthRenderer, fifthRenderer ) {
    firstRenderer = firstRenderer || 0;
    secondRenderer = secondRenderer || 0;
    thirdRenderer = thirdRenderer || 0;
    fourthRenderer = fourthRenderer || 0;
    fifthRenderer = fifthRenderer || 0;

    // uses 25 bits now with 5 renderers
    return firstRenderer |
           ( secondRenderer << 5 ) |
           ( thirdRenderer << 10 ) |
           ( fourthRenderer << 15 ) |
           ( fifthRenderer << 20 );
  };
  Renderer.bitmaskOrderFirst = function( bitmask ) {
    return bitmask & 0x000001F;
  };
  Renderer.bitmaskOrderSecond = function( bitmask ) {
    return ( bitmask >> 5 ) & 0x000001F;
  };
  Renderer.bitmaskOrderThird = function( bitmask ) {
    return ( bitmask >> 10 ) & 0x000001F;
  };
  Renderer.bitmaskOrderFourth = function( bitmask ) {
    return ( bitmask >> 15 ) & 0x000001F;
  };
  Renderer.bitmaskOrderFifth = function( bitmask ) {
    return ( bitmask >> 20 ) & 0x000001F;
  };
  Renderer.pushOrderBitmask = function( bitmask, renderer ) {
    assert && assert( typeof bitmask === 'number' );
    assert && assert( typeof renderer === 'number' );
    var rendererToInsert = renderer;
    for ( var i = 0; i < 30; i += 5 ) {
      var currentRenderer = ( bitmask >> i ) & 0x000001F;
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
        bitmask = ( bitmask & ~( 0x000001F << i ) );

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
    else if ( Renderer.isWebGL( selfRenderer ) ) {
      return node.createWebGLDrawable( selfRenderer, instance );
    }
    else if ( Renderer.isPixi( selfRenderer ) ) {
      return node.createPixiDrawable( selfRenderer, instance );
    }
    else {
      throw new Error( 'Unrecognized renderer: ' + selfRenderer );
    }
  };

  /*---------------------------------------------------------------------------*
  * WebGL Renderer type enumeration
  *----------------------------------------------------------------------------*/
  Renderer.webglCustom = 0x1;
  Renderer.webglTexturedTriangles = 0x2;

  return Renderer;
} );
