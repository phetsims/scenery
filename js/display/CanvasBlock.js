// Copyright 2002-2014, University of Colorado

/**
 * Handles a visual Canvas layer of drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var scenery = require( 'SCENERY/scenery' );
  var Block = require( 'SCENERY/display/Block' );
  var CanvasContextWrapper = require( 'SCENERY/util/CanvasContextWrapper' );
  
  scenery.CanvasBlock = function CanvasBlock( display, renderer, transformRootInstance ) {
    this.initialize( display, renderer, transformRootInstance );
  };
  var CanvasBlock = scenery.CanvasBlock;
  
  inherit( Block, CanvasBlock, {
    initialize: function( display, renderer, transformRootInstance ) {
      this.initializeBlock( display, renderer );
      
      this.transformRootInstance = transformRootInstance;
      
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );
      
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
      
      return this;
    },
    
    dispose: function() {
      // clear references
      this.transformRootInstance = null;
      cleanArray( this.dirtyDrawables );
      
      // minimize memory exposure of the backing raster
      this.canvas.width = 0;
      this.canvas.height = 0;
      
      Block.prototype.dispose.call( this );
    },
    
    update: function() {
      if ( this.dirty && !this.disposed ) {
        this.dirty = false;
        
        while ( this.dirtyDrawables.length ) {
          this.dirtyDrawables.pop().update();
        }
        
        // TODO: repaint here
      }
    },
    
    markDirtyDrawable: function( drawable ) {
      assert && assert( drawable );
      
      // TODO: instance check to see if it is a canvas cache (usually we don't need to call update on our drawables)
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },
    
    addDrawable: function( drawable ) {
      Block.prototype.addDrawable.call( this, drawable );
    },
    
    removeDrawable: function( drawable ) {
      Block.prototype.removeDrawable.call( this, drawable );
    }
  } );
  
  /* jshint -W064 */
  Poolable( CanvasBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, renderer, transformRootInstance ) {
        if ( pool.length ) {
          return pool.pop().initialize( display, renderer, transformRootInstance );
        } else {
          return new CanvasBlock( display, renderer, transformRootInstance );
        }
      };
    }
  } );
  
  return CanvasBlock;
} );
