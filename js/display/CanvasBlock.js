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
  var CanvasContextWrapper = require( 'SCENERY/util/CanvasContextWrapper' );
  
  scenery.CanvasBlock = function CanvasBlock( renderer, transformRootInstance ) {
    this.initialize( renderer, transformRootInstance );
  };
  var CanvasBlock = scenery.CanvasBlock;
  
  inherit( Drawable, CanvasBlock, {
    initialize: function( renderer, transformRootInstance ) {
      this.initializeDrawable( renderer );
      
      this.transformRootInstance = transformRootInstance;
      
      if ( !this.domElement ) {
        //OHTWO TODO: support tiled Canvas handling (will need to wrap then in a div, or something)
        this.canvas = document.createElement( 'canvas' );
        this.context = this.canvas.getContext( '2d' );
        
        // workaround for Chrome (WebKit) miterLimit bug: https://bugs.webkit.org/show_bug.cgi?id=108763
        this.context.miterLimit = 20;
        this.context.miterLimit = 10;
        
        this.wrapper = new CanvasContextWrapper( this.canvas, this.context );
        
        this.domElement = this.canvas;
      }
      
      // TODO: add count of boundsless objects?
      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)
    },
    
    dispose: function() {
      // clear references
      this.transformRootInstance = null;
      
      // minimize memory exposure of the backing raster
      this.canvas.width = 0;
      this.canvas.height = 0;
      
      Drawable.prototype.dispose.call( this );
    },
    
    markDirtyDrawable: function( drawable ) {
      
    },
    
    addDrawable: function( drawable ) {
      
    },
    
    removeDrawable: function( drawable ) {
      
    }
  } );
  
  /* jshint -W064 */
  Poolable( CanvasBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( renderer, transformRootInstance ) {
        if ( pool.length ) {
          return pool.pop().initialize( renderer, transformRootInstance );
        } else {
          return new CanvasBlock( renderer, transformRootInstance );
        }
      };
    }
  } );
  
  return CanvasBlock;
} );
