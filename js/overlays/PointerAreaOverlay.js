// Copyright 2002-2013, University of Colorado Boulder

/**
 * Displays mouse and touch areas when they are customized. Expensive to display!
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var Bounds2 = require( 'DOT/Bounds2' );
  var Transform3 = require( 'DOT/Transform3' );
  var Matrix3 = require( 'DOT/Matrix3' );

  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );

  var Util = require( 'SCENERY/util/Util' );

  scenery.PointerAreaOverlay = function PointerAreaOverlay( scene ) {
    this.scene = scene;
    
    var svg = this.svg = document.createElementNS( 'http://www.w3.org/2000/svg', 'svg' );
    svg.style.position = 'absolute';
    svg.className = 'mouseTouchAreaOverlay';
    svg.style.top = 0;
    svg.style.left = 0;
    svg.style['pointer-events'] = 'none';
    
    function resize( width, height ) {
      svg.setAttribute( 'width', width );
      svg.setAttribute( 'height', height );
      svg.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
    }
    scene.addEventListener( 'resize', function( args ) {
      resize( args.width, args.height );
    } );
    resize( scene.getSceneWidth(), scene.getSceneHeight() );
    
    scene.$main[0].appendChild( svg );
    
    scene.reindexLayers();
  };
  var PointerAreaOverlay = scenery.PointerAreaOverlay;

  PointerAreaOverlay.prototype = {
    dispose: function() {
      this.scene.$main[0].removeChild( this.svg );
    },

    reindex: function( index ) {
      this.svg.style.zIndex = index;
    },
    
    addShape: function( shape, color, isOffset ) {
      var path = document.createElementNS( 'http://www.w3.org/2000/svg', 'path' );
      var svgPath = shape.getSVGPath();
      
      // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
      // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
      // the attribute can cause this bug
      if ( !svgPath ) { svgPath = 'M0 0'; }
      
      if ( svgPath ) {
        // only set the SVG path if it's not the empty string
        path.setAttribute( 'd', svgPath );
      } else if ( path.hasAttribute( 'd' ) ) {
        path.removeAttribute( 'd' );
      }
      
      path.setAttribute( 'style', 'fill: none; stroke: ' + color + '; stroke-dasharray: 5, 3; stroke-dashoffset: ' + ( isOffset ? 5 : 0 ) + '; stroke-width: 3;' );
      this.svg.appendChild( path );
    },
    
    update: function() {
      var that = this;
      var svg = this.svg;
      var scene = this.scene;
      
      while ( svg.childNodes.length ) {
        svg.removeChild( svg.childNodes[svg.childNodes.length-1] );
      }
      
      new scenery.Trail( scene ).eachTrailUnder( function( trail ) {
        var node = trail.lastNode();
        if ( node._mouseArea || node._touchArea ) {
          var transform = trail.getTransform();
          
          if ( node._mouseArea ) {
            that.addShape( transform.transformShape( node._mouseArea ), 'rgba(0,0,255,0.8)', true );
          }
          if ( node._touchArea ) {
            that.addShape( transform.transformShape( node._touchArea ), 'rgba(255,0,0,0.8)', false );
          }
        }
      } );
    }
  };

  return PointerAreaOverlay;
} );
