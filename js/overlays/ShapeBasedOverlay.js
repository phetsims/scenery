// Copyright 2013-2015, University of Colorado Boulder

/**
 * Supertype for overlays that display colored shapes (updated every frame).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );

  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );

  function ShapeBasedOverlay( display, rootNode, name ) {
    this.display = display;
    this.rootNode = rootNode;

    var svg = this.svg = document.createElementNS( scenery.svgns, 'svg' );
    svg.style.position = 'absolute';
    svg.setAttribute( 'class', name );
    svg.style.top = 0;
    svg.style.left = 0;
    svg.style[ 'pointer-events' ] = 'none';

    function resize( width, height ) {
      svg.setAttribute( 'width', width );
      svg.setAttribute( 'height', height );
      svg.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
    }

    display.onStatic( 'displaySize', function( dimension ) {
      resize( dimension.width, dimension.height );
    } );
    resize( display.width, display.height );

    this.domElement = svg;
  }

  scenery.register( 'ShapeBasedOverlay', ShapeBasedOverlay );

  inherit( Object, ShapeBasedOverlay, {
    addShape: function( shape, color, isOffset ) {
      var path = document.createElementNS( scenery.svgns, 'path' );
      var svgPath = shape.getSVGPath();

      // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
      // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
      // the attribute can cause this bug
      if ( !svgPath ) { svgPath = 'M0 0'; }

      if ( svgPath ) {
        // only set the SVG path if it's not the empty string
        path.setAttribute( 'd', svgPath );
      }
      else if ( path.hasAttribute( 'd' ) ) {
        path.removeAttribute( 'd' );
      }

      path.setAttribute( 'style', 'fill: none; stroke: ' + color + '; stroke-dasharray: 5, 3; stroke-dashoffset: ' + ( isOffset ? 5 : 0 ) + '; stroke-width: 3;' );
      this.svg.appendChild( path );
    },

    update: function() {
      while ( this.svg.childNodes.length ) {
        this.svg.removeChild( this.svg.childNodes[ this.svg.childNodes.length - 1 ] );
      }

      this.addShapes();
    },

    // STUB to be overridden
    addShapes: function() {

    },

    dispose: function() {

    }
  } );

  return ShapeBasedOverlay;
} );
