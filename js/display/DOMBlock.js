// Copyright 2002-2014, University of Colorado

/**
 * DOM Drawable wrapper for another DOM Drawable. Used so that we can have our own independent siblings, generally as part
 * of a Backbone's layers/blocks.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.DOMBlock = function DOMBlock( display, domDrawable ) {
    this.initialize( display, domDrawable );
  };
  var DOMBlock = scenery.DOMBlock;
  
  inherit( Drawable, DOMBlock, {
    initialize: function( display, domDrawable ) {
      // TODO: is it bad to pass the acceleration flags along?
      this.initializeDrawable( domDrawable.renderer );
      
      this.display = display;
      this.domDrawable = domDrawable;
      this.domElement = domDrawable.domElement;
      
      return this;
    },
    
    dispose: function() {
      this.domDrawable = null;
      this.domElement = null;
      this.display = null;
      
      // super call
      Drawable.prototype.dispose.call( this );
    },
    
    update: function() {
      if ( this.dirty && !this.disposed ) {
        this.dirty = false;
        
        this.domDrawable.update();
      }
    },
    
    markDirtyDrawable: function( drawable ) {
      this.markDirty();
    },
    
    addDrawable: function( drawable ) {
      assert && assert( this.domDrawable === drawable, 'DOMBlock should only be used with one drawable for now (the one it was initialized with)' );
      drawable.parentDrawable = this;
    },
    
    removeDrawable: function( drawable ) {
      assert && assert( this.domDrawable === drawable, 'DOMBlock should only be used with one drawable for now (the one it was initialized with)' );
      drawable.parentDrawable = null;
    }
  } );

  /* jshint -W064 */
  Poolable( DOMBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, domDrawable ) {
        if ( pool.length ) {
          return pool.pop().initialize( display, domDrawable );
        } else {
          return new DOMBlock( display, domDrawable );
        }
      };
    }
  } );
  
  return DOMBlock;
} );
