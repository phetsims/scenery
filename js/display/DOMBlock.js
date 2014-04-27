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
  
  scenery.DOMBlock = function DOMBlock( domDrawable ) {
    this.initialize( domDrawable );
  };
  var DOMBlock = scenery.DOMBlock;
  
  inherit( Drawable, DOMBlock, {
    initialize: function( domDrawable ) {
      // TODO: is it bad to pass the acceleration flags along?
      this.initializeDrawable( domDrawable.renderer );
      
      this.domDrawable = domDrawable;
      this.domElement = domDrawable.domElement;
      
      return this;
    },
    
    dispose: function() {
      this.domDrawable = null;
      this.domElement = null;
      
      // super call
      Drawable.prototype.dispose.call( this );
    },
    
    addDrawable: function( drawable ) {
      assert && assert( this.domDrawable === drawable, 'DOMBlock should only be used with one drawable for now (the one it was initialized with)' );
    },
    
    removeDrawable: function( drawable ) {
      assert && assert( this.domDrawable === drawable, 'DOMBlock should only be used with one drawable for now (the one it was initialized with)' );
    }
  } );

  /* jshint -W064 */
  Poolable( DOMBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( domDrawable ) {
        if ( pool.length ) {
          return pool.pop().initialize( domDrawable );
        } else {
          return new DOMBlock( domDrawable );
        }
      };
    }
  } );
  
  return DOMBlock;
} );
