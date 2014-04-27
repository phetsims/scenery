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
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.DOMWrapperDrawable = function DOMWrapperDrawable( domDrawable ) {
    this.initialize( domDrawable );
  };
  var DOMWrapperDrawable = scenery.DOMWrapperDrawable;
  
  inherit( SelfDrawable, DOMWrapperDrawable, {
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
    }
  } );

  /* jshint -W064 */
  Poolable( DOMWrapperDrawable, {
    constructorDuplicateFactory: function( pool ) {
      return function( domDrawable ) {
        if ( pool.length ) {
          return pool.pop().initialize( domDrawable );
        } else {
          return new DOMWrapperDrawable( domDrawable );
        }
      };
    }
  } );
  
  return DOMWrapperDrawable;
} );
