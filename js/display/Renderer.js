// Copyright 2013-2016, University of Colorado Boulder


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
  var Renderer = {};
  scenery.register( 'Renderer', Renderer );

  //OHTWO TODO: rename to take advantage of lack of deprecated names? (remove bitmask prefix)

  /*---------------------------------------------------------------------------*
   * Renderer bitmask flags
   *---------------------------------------------------------------------------*/

  Renderer.numActiveRenderers = 4;
  Renderer.bitsPerRenderer = 5;
  Renderer.bitmaskRendererArea = 0x00000FF;
  Renderer.bitmaskCurrentRendererArea = 0x000000F;
  Renderer.bitmaskLacksOffset = 0x10000;
  Renderer.bitmaskLacksShift = 16; // number of bits between the main renderer bitmask and the "lacks" variety
  Renderer.bitmaskNodeDefault = Renderer.bitmaskRendererArea;

  Renderer.bitmaskCanvas = 0x0000001;
  Renderer.bitmaskSVG = 0x0000002;
  Renderer.bitmaskDOM = 0x0000004;
  Renderer.bitmaskWebGL = 0x0000008;
  // 10, 20, 40, 80 reserved for future renderers NOTE: update bitmaskCurrentRendererArea/numActiveRenderers if they are added/removed

  // summary bits (for RendererSummary):
  Renderer.bitmaskSingleCanvas = 0x100;
  Renderer.bitmaskSingleSVG = 0x200;
  // reserved gap 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000 for future renderer-specific single information
  Renderer.bitmaskNotPainted = 0x1000;
  Renderer.bitmaskBoundsValid = 0x2000;
  Renderer.bitmaskNotAccessible = 0x4000;
  // summary bits for whether a renderer could be potentially used to display a Node.
  Renderer.bitmaskLacksCanvas = Renderer.bitmaskCanvas << Renderer.bitmaskLacksShift; // 0x10000
  Renderer.bitmaskLacksSVG = Renderer.bitmaskSVG << Renderer.bitmaskLacksShift; // 0x20000
  Renderer.bitmaskLacksDOM = Renderer.bitmaskDOM << Renderer.bitmaskLacksShift; // 0x40000
  Renderer.bitmaskLacksWebGL = Renderer.bitmaskWebGL << Renderer.bitmaskLacksShift; // 0x80000
  // reserved gap 0x10000, 0x20000, 0x40000, 0x80000 for future renderers

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

  var rendererMap = {
    canvas: Renderer.bitmaskCanvas,
    svg: Renderer.bitmaskSVG,
    dom: Renderer.bitmaskDOM,
    webgl: Renderer.bitmaskWebGL
  };
  Renderer.fromName = function( name ) {
    return rendererMap[ name ];
  };

  // returns the part of the bitmask that should contain only Canvas/SVG/DOM/WebGL flags
  //OHTWO TODO: use this instead of direct access to bitmaskRendererArea
  Renderer.stripBitmask = function( bitmask ) {
    return bitmask & Renderer.bitmaskRendererArea;
  };

  Renderer.createOrderBitmask = function( firstRenderer, secondRenderer, thirdRenderer, fourthRenderer ) {
    firstRenderer = firstRenderer || 0;
    secondRenderer = secondRenderer || 0;
    thirdRenderer = thirdRenderer || 0;
    fourthRenderer = fourthRenderer || 0;

    // uses 20 bits now with 4 renderers
    return firstRenderer |
           ( secondRenderer << 5 ) |
           ( thirdRenderer << 10 ) |
           ( fourthRenderer << 15 );
  };
  // bitmaskOrderN with n=0 is bitmaskOrderFirst, n=1 is bitmaskOrderSecond, etc.
  Renderer.bitmaskOrder = function( bitmask, n ) {
    // Normally the condition here shouldn't be needed, but Safari seemed to cause a logic error when this function
    // gets inlined elsewhere if n=0. See https://github.com/phetsims/scenery/issues/481 and
    // https://github.com/phetsims/bending-light/issues/259.
    if ( n > 0 ) {
      bitmask = bitmask >> ( 5 * n );
    }
    return bitmask & Renderer.bitmaskCurrentRendererArea;
  };
  Renderer.bitmaskOrderFirst = function( bitmask ) {
    return bitmask & Renderer.bitmaskCurrentRendererArea;
  };
  Renderer.bitmaskOrderSecond = function( bitmask ) {
    return ( bitmask >> 5 ) & Renderer.bitmaskCurrentRendererArea;
  };
  Renderer.bitmaskOrderThird = function( bitmask ) {
    return ( bitmask >> 10 ) & Renderer.bitmaskCurrentRendererArea;
  };
  Renderer.bitmaskOrderFourth = function( bitmask ) {
    return ( bitmask >> 15 ) & Renderer.bitmaskCurrentRendererArea;
  };
  Renderer.pushOrderBitmask = function( bitmask, renderer ) {
    assert && assert( typeof bitmask === 'number' );
    assert && assert( typeof renderer === 'number' );
    var rendererToInsert = renderer;
    var totalBits = Renderer.bitsPerRenderer * Renderer.numActiveRenderers;
    for ( var i = 0; i <= totalBits; i += Renderer.bitsPerRenderer ) {
      var currentRenderer = ( bitmask >> i ) & Renderer.bitmaskCurrentRendererArea;
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
        bitmask = ( bitmask & ~( Renderer.bitmaskCurrentRendererArea << i ) );

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

  Renderer.createSelfDrawable = function( instance, node, selfRenderer, fittable ) {
    var drawable;

    if ( Renderer.isCanvas( selfRenderer ) ) {
      drawable = node.createCanvasDrawable( selfRenderer, instance );
    }
    else if ( Renderer.isSVG( selfRenderer ) ) {
      drawable = node.createSVGDrawable( selfRenderer, instance );
    }
    else if ( Renderer.isDOM( selfRenderer ) ) {
      drawable = node.createDOMDrawable( selfRenderer, instance );
    }
    else if ( Renderer.isWebGL( selfRenderer ) ) {
      drawable = node.createWebGLDrawable( selfRenderer, instance );
    }
    else {
      throw new Error( 'Unrecognized renderer: ' + selfRenderer );
    }

    // Check to make sure that all of the drawables have the required mark-dirty methods available.
    if ( assert ) {
      _.each( node.drawableMarkFlags, function( flag ) {
        var methodName = 'markDirty' + flag[ 0 ].toUpperCase() + flag.slice( 1 );
        assert( typeof drawable[ methodName ] === 'function', 'Did not find ' + methodName );
      } );
    }

    // Initialize its fittable flag
    drawable.setFittable( fittable );

    return drawable;
  };

  /*---------------------------------------------------------------------------*
   * WebGL Renderer type enumeration
   *----------------------------------------------------------------------------*/
  Renderer.webglCustom = 0x1;
  Renderer.webglTexturedTriangles = 0x2;
  Renderer.webglVertexColorPolygons = 0x3;

  return Renderer;
} );
