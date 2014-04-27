// Copyright 2002-2014, University of Colorado

/**
 * TODO docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.SVGBlock = function SVGBlock( renderer, transformRootInstance, filterRootInstance ) {
    this.initialize( renderer, transformRootInstance, filterRootInstance );
  };
  var SVGBlock = scenery.SVGBlock;
  
  inherit( Drawable, SVGBlock, {
    initialize: function( renderer, transformRootInstance, filterRootInstance ) {
      this.initializeDrawable( renderer );
      
      this.transformRootInstance = transformRootInstance;
      this.filterRootInstance = filterRootInstance;
      
      if ( !this.domElement ) {
        // main SVG element
        this.svg = document.createElementNS( scenery.svgns, 'svg' );
        // this.svg.setAttribute( 'width', width );
        // this.svg.setAttribute( 'height', height );
        this.svg.setAttribute( 'stroke-miterlimit', 10 ); // to match our Canvas brethren so we have the same default behavior
        // this.svg.style.position = 'absolute';
        // this.svg.style.left = '0';
        // this.svg.style.top = '0';
        // this.svg.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
        this.svg.style['pointer-events'] = 'none';
        
        // the <defs> block that we will be stuffing gradients and patterns into
        this.defs = document.createElementNS( scenery.svgns, 'defs' );
        this.svg.appendChild( this.defs );
        
        this.domElement = this.svg;
      }
      
      // TODO: add count of boundsless objects?
      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)
    },
    
    dispose: function() {
      // clear references
      this.transformRootInstance = null;
      this.filterRootInstance = null;
      
      Drawable.prototype.dispose.call( this );
    },
    
    markDirtyDrawable: function( drawable ) {
      
    }
  } );
  
  /* jshint -W064 */
  Poolable( SVGBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( renderer, transformRootInstance, filterRootInstance ) {
        if ( pool.length ) {
          return pool.pop().initialize( renderer, transformRootInstance, filterRootInstance );
        } else {
          return new SVGBlock( renderer, transformRootInstance, filterRootInstance );
        }
      };
    }
  } );
  
  return SVGBlock;
} );
